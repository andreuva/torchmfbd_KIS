import importlib
import pkgutil
import numpy as np
import torch
import torch.nn.functional as F
import torchmfbd.zern as zern
import torchmfbd.util as util
from collections import OrderedDict
from tqdm import tqdm
from skimage.morphology import flood
import scipy.ndimage as nd
try:
    from nvitop import Device
    HAS_NVITOP = True
except ImportError:
    HAS_NVITOP = False
import logging
import torchmfbd.kl_modes as kl_modes
import torchmfbd.noise as noise
from torchmfbd.reg_smooth import RegularizationSmooth
from torchmfbd.reg_iuwt import RegularizationIUWT
from torchmfbd.reg_time import RegularizationTime
import glob
import pathlib
import yaml
import torchmfbd.configuration as configuration
import time
import scipy.optimize as optim
from astropy.io import fits
from einops import rearrange
from torchmfbd.vae import ConvVAE2D
import importlib.resources
try:
    import ncg_optimizer
    NGC_OPTIMIZER = True
except:
    NGC_OPTIMIZER = False
    pass

class Deconvolution(object):
    def __init__(self, config, add_piston=False):        
        """

        Parameters
        ----------
        config : str or dict
            The configuration for the deconvolution. It can be either a path to a YAML file or a dictionary containing the configuration parameters.
        add_piston : bool, optional
            Whether to include the piston mode in the wavefront expansion. The default is False.
        """
        super().__init__()

        torch.set_default_dtype(torch.float32)
        
        self.logger = logging.getLogger("deconvolution ")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        ch = logging.StreamHandler()        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.fft_norm = 'ortho'

        if isinstance(config, dict):
            self.logger.info(f"Using configuration dictionary")
            self.config = config
        else:
            self.logger.info(f"Using configuration file {config}")
            self.config = self.read_config_file(config)

        # Check configuration file for errors
        self.config = configuration._check_config(self.config)
                
        # Check the presence of a GPU
        self.cuda = torch.cuda.is_available()      

        # Check that the GPU compatible
        if self.cuda and HAS_NVITOP:
            if len(Device.all()) == 0:
                self.cuda = False  

        # Ger handlers to later check memory and usage of GPUs
        if self.cuda:
            if self.config['optimization']['gpu'] < 0:
                self.logger.info(f"GPU is available but not used. Computing in cpu")
                self.cuda = False
                self.handle = None
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{self.config['optimization']['gpu']}")
                if HAS_NVITOP:
                    self.handle = Device.all()[self.config['optimization']['gpu']]
                    self.logger.info(f"Computing in {self.handle.name()} (free {self.handle.memory_free() / 1024**3:4.2f} GB) - cuda:{self.config['optimization']['gpu']}")
                    self.initial_memory_used = self.handle.memory_used()
                else:
                    self.handle = None
                    self.logger.info(f"Computing in cuda:{self.config['optimization']['gpu']} (nvitop not installed, memory tracking disabled)")
        else:
            if not HAS_NVITOP:
                self.logger.info(f"nvitop not installed. Computing in cpu")
            else:
                self.logger.info(f"No GPU is available. Computing in cpu")
            self.device = torch.device("cpu")
            self.handle = None

        # self.n_modes = np.sum(np.arange(self.config['psf']['nmax_modes'])+2)
        self.n_modes = self.config['psf']['nmax_modes']
        self.use_jitter = self.config['psf']['jitter']
        self.npix = self.config['images']['n_pixel']
        self.npix_apod = self.config['images']['apodization_border']
        self.remove_gradient_apodization = self.config['images']['remove_gradient_apodization']

        self.psf_model = self.config['psf']['model']

        self.loss_type = self.config['optimization']['loss_type']
        
        # Whether to take into account the piston mode
        self.add_piston = add_piston                     
                
        # Generate Hamming window function for WFS correlation
        if (self.npix_apod > 0):
            self.logger.info(f"Using apodization mask with a border of {self.npix_apod} pixels")
            if self.remove_gradient_apodization:
                self.logger.info(f"Removing gradient in apodization")
            else:
                self.logger.info(f"Not removing gradient in apodization")
            win = np.hanning(2*self.npix_apod)
            winOut = np.ones(self.npix)
            winOut[0:self.npix_apod] = win[0:self.npix_apod]
            winOut[-self.npix_apod:] = win[-self.npix_apod:]
            self.window = np.outer(winOut, winOut)
        else:
            self.logger.info(f"No apodization")
            self.window = np.ones((self.npix, self.npix))

        self.window = torch.tensor(self.window.astype('float32')).to(self.device)        
                
        # Learning rates
        self.lr_obj = self.config['optimization']['lr_obj']
        self.lr_modes = self.config['optimization']['lr_modes']
        self.lr_prior = self.config['optimization']['lr_prior']
        if self.loss_type == 'marginal':
            self.stop_psd = self.config['optimization'].get('stop_psd_optimization', 1000)

        # Type of Fourier filter for the loss
        if 'filter_loss' in self.config['optimization']:
            self.loss_filter = self.config['optimization']['filter_loss']
        else:
            self.loss_filter = 'wiener'        
        self.logger.info(f"Using {self.loss_filter} filter for loss computation")

        self.enforce_positive_object = self.config['optimization']['enforce_positive_object']        
        if self.enforce_positive_object:
            self.logger.info(f"Enforcing positive object during optimization")

        # Do some output
        self.logger.info(f"Telescope")        
        self.logger.info(f"  * D: {self.config['telescope']['diameter']} m")
        self.logger.info(f"  * pix: {self.config['images']['pix_size']} arcsec")
        
        # Bookkeeping for objects and diversity
        self.ind_object = []
        self.ind_diversity = []
        self.frames = []
        self.sigma = []
        self.diversity = []

        self.external_regularizations = []

        if 'show_object_info' in self.config['optimization']:
            self.show_object_info = self.config['optimization']['show_object_info']
        else:
            self.show_object_info = False

        self.simultaneous_sequences = None
        self.infer_object = None

        if 'orthogonalize' in self.config['psf']:
            self.orthogonalize_basis = self.config['psf']['orthogonalize']
        else:
            self.orthogonalize_basis = False

        if 'remove_tt' in self.config['psf']:
            self.remove_tt = self.config['psf']['remove_tt']
        else:
            self.remove_tt = False

        if self.remove_tt:
            self.logger.info(f"Tip-tilt will be removed from the modes during optimization")

        # Per-patch diffraction cutoffs. They stay None unless set_patch_cutoffs() is called.
        # They live here and not in _define_basis(), which runs on every deconvolve().
        self.patch_cutoffs = None
        self.patch_cutoff_index = None
        self.mask_diffraction_multi = None
        self._mask_diff_batch = None

        # Per-patch number of wavefront modes. Stays None unless set_patch_modes() is called
        self.patch_n_modes = None
        self._mode_mask_batch = None

       
    def _define_basis(self, n_modes=None):

        if n_modes is not None:
            self.n_modes = n_modes

        if self.psf_model.lower() in ['zernike', 'kl']:
            
            # Get Noll's n order from the number of modes
            # The summation of the modes needs to fulfill n^2+n-2*(k+1)=0 when no piston is added
            # The summation of the modes needs to fulfill n^2+n-2*k=0 when a piston is added            
            if (self.add_piston):
                self.n_modes += 1
                a = 1.0
                b = 1.0
                c = -2.0 * self.n_modes
            else:            
                a = 1.0
                b = 1.0
                c = -2.0 * (self.n_modes + 1)            
            
            sol1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
            sol2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
            
            n = 0        
            if sol1 > 0.0 and sol1.is_integer():
                n = int(sol1)
            if sol2 > 0.0 and sol2.is_integer():
                n = int(sol2)
            
            if n == 0:
                if sol1 > 0.0:
                    sol = np.floor(sol1)
                if sol2 > 0.0:
                    sol = np.floor(sol2)
                
                if (self.add_piston):
                    k_sol = (sol**2 + sol) / 2.0                    
                else:
                    k_sol = (sol**2 + sol) / 2.0 - 1
                                        
                raise Exception(f"Number of modes {self.n_modes} do not cover a full radial degree. Closest value : {k_sol}")

            self.noll_max = n   

        self.pupil = [None] * self.n_o
        self.basis = [None] * self.n_o
        self.defocus_basis = [None] * self.n_o
        self.rho = [None] * self.n_o
        self.f_x = [None] * self.n_o
        self.f_y = [None] * self.n_o
        self.s_u = [None] * self.n_o
        self.diffraction_limit = [None] * self.n_o
        self.cutoff = [None] * self.n_o
        self.image_filter = [None] * self.n_o

        # First locate unique wavelengths. We will use the same basis for the same wavelength
        ind_wavelengths = []
        unique_wavelengths = []

        # Options of the Fourier noise filter applied to the object estimate
        self.filter_mode = [None] * self.n_o
        self.filter_threshold = [None] * self.n_o
        self.filter_noise_band = [None] * self.n_o
        self.filter_noise_shape = [None] * self.n_o
        self.filter_noise_flat = [None] * self.n_o
        self.filter_weight = [None] * self.n_o

        for i in range(self.n_o):
            self.cutoff[i] = self.config[f'object{i+1}']['cutoff']
            self.image_filter[i] = self.config[f'object{i+1}']['image_filter']
            self.s_u[i] = self.config[f'object{i+1}']['s_u_joint']
            self.filter_mode[i] = self.config[f'object{i+1}']['filter_mode']
            self.filter_threshold[i] = self.config[f'object{i+1}']['filter_threshold']
            self.filter_noise_band[i] = self.config[f'object{i+1}']['filter_noise_band']
            self.filter_noise_shape[i] = self.config[f'object{i+1}']['filter_noise_shape']
            self.filter_noise_flat[i] = self.config[f'object{i+1}']['filter_noise_flat']
            w = self.config[f'object{i+1}']['wavelength']
            if w not in unique_wavelengths:
                unique_wavelengths.append(w)
            
            ind_wavelengths.append(unique_wavelengths.index(w))

        # Normalize wavelengths to scale basis
        unique_wavelengths = np.array(unique_wavelengths).astype('float32')
        normalized_wavelengths = unique_wavelengths / np.max(unique_wavelengths)

        # Now iterate over all unique wavelengths and associate the basis
        # to the corresponding object
        for i in range(len(unique_wavelengths)):

            wavelength = unique_wavelengths[i]

            # Compute the overfill to properly generate the PSFs from the wavefronts
            overfill = util.psf_scale(wavelength, 
                                    self.config['telescope']['diameter'], 
                                    self.config['images']['pix_size'])

            if (overfill < 1.0):
                raise Exception(f"The pixel size is not small enough to model a telescope with D={self.config['telescope']['diameter']} cm")

            # Compute telescope aperture
            pixel_size_pupil = self.config['telescope']['diameter'] / self.npix
            pupil = util.aperture(npix=self.npix, 
                            cent_obs = self.config['telescope']['central_obscuration'] / self.config['telescope']['diameter'], 
                            spider=self.config['telescope']['spider'] / pixel_size_pupil, 
                            overfill=overfill)
            
            # Obtain defocus basis for phase diversity
            defocus_basis = self.get_defocus_basis(overfill=overfill)
                        
            # PSF model parameterized with the wavefront
            if (self.psf_model.lower() in ['zernike', 'kl', 'nmf']):
                            
                if (self.psf_model.lower() not in ['zernike', 'kl', 'nmf']):
                    raise Exception(f"Unknown basis {self.basis}. Use 'zernike' or 'kl' for wavefront expansion")
            
                if (self.psf_model.lower() == 'zernike'):

                    self.logger.info(f"PSF model: wavefront expansion in Zernike modes")
                    
                    found, filename = self.find_basis_wavefront('zernike', self.n_modes, int(wavelength))

                    # Define Zernike modes        
                    if found:
                        self.logger.info(f"Loading precomputed Zernike {filename}")
                        tmp = np.load(f"{filename}")
                        basis = tmp['basis']           
                    else:                
                        self.logger.info(f"Computing Zernike modes {filename}")
                        basis = self.precalculate_zernike(overfill=overfill)

                        # Add piston mode if needed
                        if (self.add_piston):
                            basis = np.concatenate([pupil * np.ones((1, self.npix, self.npix)), basis[0:self.n_modes, :, :]], axis=0)
                            self.n_modes += 1

                        # Orthogonalize the Zernike modes if needed
                        if self.orthogonalize_basis:
                            self.logger.info(f"Orthogonalizing Zernike modes")
                            basis = util.orthogonalize(basis, pupil)
                            self.logger.info(f"  * Orthogonalization done")

                        np.savez(f"{filename}", basis=basis)

                if (self.psf_model.lower() == 'kl'):

                    self.logger.info(f"PSF model: wavefront expansion in KL modes")

                    found, filename = self.find_basis_wavefront('kl', self.n_modes, int(wavelength))

                    if found:
                        self.logger.info(f"Loading precomputed KL basis: {filename}")
                        tmp = np.load(f"{filename}")
                        basis = tmp['basis']                
                    else:
                        self.logger.info(f"Computing KL modes {filename}")
                        self.kl = kl_modes.KL()
                        
                        basis = self.kl.precalculate(npix_image = self.npix, 
                                        n_modes_max = self.n_modes,                                 
                                        overfill=overfill)
                        
                        # Add piston mode if needed                        
                        if (self.add_piston):
                            basis = np.concatenate([pupil * np.ones((1, self.npix, self.npix)), basis[0:self.n_modes, :, :]], axis=0)
                            self.n_modes += 1

                        # Orthogonalize the KL modes if needed
                        if self.orthogonalize_basis:
                            self.logger.info(f"Orthogonalizing KL modes")
                            basis = util.orthogonalize(basis, pupil)
                            self.logger.info(f"  * Orthogonalization done")

                        np.savez(f"{filename}", basis=basis, variance=self.kl.varKL)
                
                if (self.psf_model.lower() == 'nmf'):
                    
                    self.logger.info(f"PSF model: PSF expansion in NMF modes")

                    self.logger.info(f"Loading precomputed NMF basis: {self.config['psf']['filename']}")
                    f = np.load(self.config['psf']['filename'])

                    n_psf = int(np.sqrt(f['basis'].shape[1]))

                    basis = f['basis'][0:self.n_modes, :].reshape((self.n_modes, n_psf, n_psf))

                    basis = basis[:, n_psf//2 - self.npix//2:n_psf//2 + self.npix//2, n_psf//2 - self.npix//2:n_psf//2 + self.npix//2]
                                    
                pupil = torch.tensor(pupil.astype('float32')).to(self.device)
                basis = torch.tensor(basis[0:self.n_modes, :, :].astype('float32')).to(self.device)
                defocus_basis = torch.tensor(defocus_basis.astype('float32')).to(self.device)

                # Following van Noort et al. (2005) we normalize the basis to the maximum wavelength
                # so that the wavefront is given in radians
                if (self.psf_model.lower() in ['zernike', 'kl']):
                    basis /= normalized_wavelengths[i]
                    defocus_basis /= normalized_wavelengths[i]
                elif self.psf_model.lower() == 'nmf':
                    # Read the exponent from the YAML config (defaults to 1.0)
                    basis_power = self.config['psf'].get('basis_power', 1.0)
                    
                    if basis_power != 1.0:
                        self.logger.info(f"  * Applying 'fat wing' power: {basis_power}")
                        # Apply power (with epsilon to prevent numerical issues at exactly zero)
                        basis = torch.pow(basis + 1e-12, basis_power)
                    
                    # Normalize each NMF basis function to have unit area
                    basis = basis / (torch.sum(basis, dim=[-1, -2], keepdim=True) + 1e-12)

                if (self.add_piston):
                    self.logger.info(f"Adding piston mode...")
                
                self.logger.info(f"  * Using {self.n_modes} modes...")

            if (self.psf_model.lower() == 'vae'):                
                self.logger.info(f"PSF model: PSF expansion in VAE modes")
               
# Get a Traversable path object to the file
                filename = importlib.resources.files("torchmfbd").joinpath("vae", "nonlinear_matryoshka_z_200_beta_0.1.pth")
                
                checkpoint = torch.load(filename, map_location=lambda storage, loc: storage, weights_only=False)

                self.x_vae_mean = torch.tensor(checkpoint['x_mean'].astype('float32')).to(self.device)
                self.x_vae_std = torch.tensor(checkpoint['x_std'].astype('float32')).to(self.device)
                latent_dim = checkpoint['latent_dim']
                n_pixel = checkpoint['n_pixel']
                nested_dims = checkpoint['nested_dims']

                # Model and Optimizer initialization
                self.model_vae = ConvVAE2D(in_channels=1, img_size=n_pixel, latent_dim=latent_dim, nested_dims=nested_dims).to(self.device)

                self.model_vae.load_state_dict(checkpoint['state_dict'])
                self.model_vae.eval()

                self.logger.info(f"VAE is trained for PSFs of size {n_pixel}x{n_pixel} pixels with latent dimension {latent_dim}")
                if n_pixel > self.npix:
                    self.logger.info(f"PSFs will be cropped to generate PSFs of size {self.npix}x{self.npix} pixels")
                if n_pixel < self.npix:
                    self.logger.info(f"PSFs will be padded to generate PSFs of size {self.npix}x{self.npix} pixels")
                
                self.logger.info(f"Number of VAE model parameters: {sum(p.numel() for p in self.model_vae.parameters())}")
            
            # Output jitter
            if self.use_jitter == 'slow' or self.use_jitter == 'fast':
                self.logger.info(f"PSF model: including {self.use_jitter} jitter")
                
            # Compute the diffraction limit and the frequency grid
            # Frequency is given in units of the cutoff frequency
            cutoff = self.config['telescope']['diameter'] / (wavelength * 1e-8) / 206265.0
            freq = np.fft.fftfreq(self.npix, d=self.config['images']['pix_size']) / cutoff
            
            f_x, f_y = np.meshgrid(freq, freq)
            rho = np.sqrt(f_x ** 2 + f_y ** 2).astype('float32')
            rho = torch.tensor(rho.astype('float32')).to(self.device)

            # Fourier coordinates for jitter OTF given in pixel
            f = np.fft.fftfreq(self.npix)
            f_x, f_y = np.meshgrid(f, f)
            f_x = torch.tensor(f_x.astype('float32')).to(self.device)
            f_y = torch.tensor(f_y.astype('float32')).to(self.device)

            x = np.linspace(-1, 1, self.npix)
            y = np.linspace(-1, 1, self.npix)            
            self.X = torch.tensor(x.astype('float32')).to(self.device)
            self.Y = torch.tensor(y.astype('float32')).to(self.device)
            self.X = self.X.view(1, 1, self.npix, 1)            
            self.Y = self.Y.view(1, 1, 1, self.npix)
            self.K = 50
            self.sigma_alias = 0.02
            self.tau = torch.linspace(0, 1, self.K).to(self.device)
            
            diffraction_limit = wavelength * 1e-8 / self.config['telescope']['diameter'] * 206265.0

            self.logger.info(f"Wavelength {i} ({wavelength} A)")
            self.logger.info(f"  * Diffraction: {diffraction_limit} arcsec")
            self.logger.info(f"  * Diffraction (x1.22): {1.22 * diffraction_limit} arcsec")

            for j in range(self.n_o):
                if ind_wavelengths[j] == i:

                    if self.psf_model.lower() in ['zernike', 'kl', 'nmf']:                        
                        self.pupil[j] = pupil
                        self.basis[j] = basis
                        self.defocus_basis[j] = defocus_basis
                    self.rho[j] = rho
                    self.f_x[j] = f_x
                    self.f_y[j] = f_y
                    self.diffraction_limit[j] = diffraction_limit

        return
    
    def set_regularizations(self):

        # Regularization parameters
        self.logger.info(f"Regularizations")
        self.regularization = []
        self.index_regularization = {
            'tiptilt': [],
            'modes': [],
            'object': []
        }
        
        loop = 0

        for k, v in self.config['regularization'].items():            
            if v['lambda'] != 0.0:
                
                if 'smooth' in k:                
                    tmp = RegularizationSmooth(lambda_reg=v['lambda'], variable=v['variable'])
                if 'iuwt' in k:
                    tmp = RegularizationIUWT(lambda_reg=v['lambda'], variable=v['variable'], nbands=v['nbands'], n_pixel=self.npix)
                if 'time' in k:                    
                    self.time_regularization = float(v['lambda'])

                if 'time' not in k:
                    self.regularization.append(tmp.to(self.device))
                    self.index_regularization[v['variable']].append(loop)
                    tmp.print()
                    loop += 1

        self.logger.info(f"External regularizations")
        for reg in self.external_regularizations:
            self.regularization.append(reg.to(self.device))
            self.index_regularization[reg.variable].append(loop)
            reg.print()
            loop += 1

    def add_external_regularizations(self, external_regularization):        
        """
        Adds external regularizations to the model.

        Parameters:
        -----------

        external_regularization : callable
            A function or callable object that applies the external regularization.
        lambda_reg : float
            The regularization strength parameter.
        variable : str
            The name of the variable to which the regularization is applied.
        **kwargs : dict
            Additional keyword arguments to pass to the external_regularization function.

        """

        # Regularization parameters
        self.logger.info(f"External regularization")
        
        self.external_regularizations.append(external_regularization)
        
    def read_config_file(self, filename):
        """
        Read a configuration file in YAML format.

        Parameters:
        -----------
        filename : str
            The name of the configuration file.

        Returns:
        --------
        dict
            A dictionary containing the configuration parameters.
        """

        with open(filename, 'r') as f:
            config = yaml.safe_load(f)
        
        return config

    def find_basis_wavefront(self, basis, nmax, wavelength):

        p = pathlib.Path('basis/')
        p.mkdir(parents=True, exist_ok=True)

        files = glob.glob(f"basis/{basis}_{int(self.config['telescope']['diameter'])}cm_{self.config['images']['n_pixel']}px_{wavelength}A_*.npz")

        new_file = f"basis/{basis}_{int(self.config['telescope']['diameter'])}cm_{self.npix}px_{wavelength}A_{nmax}.npz"

        if len(files) == 0:
            return False, new_file
        
        candidates = []

        for f in files:
            n = int(f.split('_')[-1].split('.')[0])
            if n >= nmax:
                candidates.append((n, f))
                self.logger.info(f"Found basis file with {n} modes that can be used for {nmax} modes")

        if len(candidates) == 0:
            return False, new_file

        # Return the file that was actually found. Rebuilding the name from the number of
        # modes alone misses the files that carry a hash of the pupil in their name, and
        # then fails to open a basis that is right there.
        n, filename = min(candidates)
        
        return True, filename
        
    def precalculate_zernike(self, overfill):
        """
        Precalculate Zernike polynomials for a given overfill factor.
        This function computes the Zernike polynomials up to `self.n_modes` and 
        returns them in a 3D numpy array. The Zernike polynomials are calculated 
        over a grid defined by `self.npix` and scaled by the `overfill` factor.
        Parameters:
        -----------
        overfill : float
            The overfill factor used to scale the radial coordinate `rho`.
        Returns:
        --------
        Z : numpy.ndarray
            A 3D array of shape (self.n_modes, self.npix, self.npix) containing 
            the precalculated Zernike polynomials. Each slice `Z[mode, :, :]` 
            corresponds to a Zernike polynomial mode.
        """
        
        Z_machine = zern.ZernikeNaive(mask=[])
        x = np.linspace(-1, 1, self.npix)
        xx, yy = np.meshgrid(x, x)
        rho = overfill * np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
        aperture_mask = rho <= 1.0

        Z = np.zeros((self.n_modes, self.npix, self.npix))

        noll_Z = 2 + np.arange(self.n_modes)
        
        for mode in tqdm(range(self.n_modes)):
                                                
            jz = noll_Z[mode]
            n, m = zern.zernIndex(jz)
            Zmode = Z_machine.Z_nm(n, m, rho, theta, True, 'Jacobi')
            Z[mode, :, :] = Zmode * aperture_mask
                
        return Z
    
    def get_defocus_basis(self, overfill):
        """
        Precalculate Zernike polynomials for a given overfill factor.
        This function computes the Zernike polynomials up to `self.n_modes` and 
        returns them in a 3D numpy array. The Zernike polynomials are calculated 
        over a grid defined by `self.npix` and scaled by the `overfill` factor.
        Parameters:
        -----------
        overfill : float
            The overfill factor used to scale the radial coordinate `rho`.
        Returns:
        --------
        Z : numpy.ndarray
            A 3D array of shape (self.n_modes, self.npix, self.npix) containing 
            the precalculated Zernike polynomials. Each slice `Z[mode, :, :]` 
            corresponds to a Zernike polynomial mode.
        """
        
        Z_machine = zern.ZernikeNaive(mask=[])
        x = np.linspace(-1, 1, self.npix)
        xx, yy = np.meshgrid(x, x)
        rho = overfill * np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
        aperture_mask = rho <= 1.0
        
        defocus = Z_machine.Z_nm(2, 0, rho, theta, True, 'Jacobi')
                
        return defocus * aperture_mask
    
    def compute_annealing(self, modes, n_iterations):
        """
        Annealing function
        We start with 2 modes and end with all modes but we give steps of the number of
        Zernike modes for each n

        Args:
            annealing (_type_): _description_
            n_iterations (_type_): _description_

        Returns:
            _type_: _description_
        """        
        
        anneal = np.zeros(n_iterations, dtype=int)
        
        # Annealing schedules          
        if self.config['annealing']['type'] == 'linear':
            self.logger.info(f"Adding modes using linear schedule")
            for i in range(n_iterations):
                if (i < self.config['annealing']['start_pct'] * n_iterations):
                    anneal[i] = modes[0]
                elif (i > self.config['annealing']['end_pct'] * n_iterations):
                    anneal[i] = modes[-1]
                else:
                    x0 = self.config['annealing']['start_pct'] * n_iterations
                    x1 = self.config['annealing']['end_pct'] * n_iterations
                    y0 = 0
                    y1 = len(modes)-1
                    index = np.clip((y1 - y0) / (x1 - x0) * (i - x0) + y0, y0, y1)
                    anneal[i] = modes[int(index)]

        if self.config['annealing']['type'] == 'sigmoid':
            self.logger.info(f"Adding modes using sigmoid schedule")
            a = 7
            b = -5
            x = np.linspace(0, 1, n_iterations)            
            anneal = (self.noll_max - 2) * (util.sigmoid(a) - util.sigmoid(a + x * (b-a)) ) / ( util.sigmoid(a) - util.sigmoid(b) )
            
            anneal = (anneal + 0.1).astype(int)            
            anneal = modes[anneal]

        if self.config['annealing']['type'] == 'none':
            self.logger.info(f"All modes always active")
            anneal = np.ones(n_iterations, dtype=int) * modes[-1]

        return anneal
    
    def compute_diffraction_masks(self):
        """
        Compute the diffraction masks for the given dimensions and store them as class attributes.
        Args:
            n_x (int): The number of pixels in the x-dimension.
            n_y (int): The number of pixels in the y-dimension.
        Attributes:
            mask_diffraction (numpy.ndarray): A 3D array of shape (n_o, n_x, n_y) containing the diffraction masks.
            mask_diffraction_th (torch.Tensor): A tensor containing the diffraction masks, converted to float32 and moved to the specified device.
            mask_diffraction_shift (numpy.ndarray): A 3D array of shape (n_o, n_x, n_y) containing the FFT-shifted diffraction masks.
        """
        
        # Compute the diffraction masks and convert to tensor
        self.mask_diffraction = [None] * self.n_o
        self.mask_diffraction_th = [None] * self.n_o
        self.mask_diffraction_shift = [None] * self.n_o
        
        for i in range(self.n_o):
            
            self.mask_diffraction[i] = torch.zeros_like(self.rho[i])
            ind = torch.where(self.rho[i] <= self.cutoff[i][0])
            self.mask_diffraction[i][ind[0], ind[1]] = 1.0

            ind = torch.where(self.rho[i] > self.cutoff[i][1])
            self.mask_diffraction[i][ind[0], ind[1]] = 0.0

            ind = torch.where((self.rho[i] > self.cutoff[i][0]) & (self.rho[i] <= self.cutoff[i][1]))

            # self.mask_diffraction[i][ind[0], ind[1]] = 1.0 - (self.rho[i][ind[0], ind[1]] - self.cutoff[i][0]) / (self.cutoff[i][1] - self.cutoff[i][0])
            self.mask_diffraction[i][ind[0], ind[1]] = torch.cos(np.pi / 2.0 * (self.rho[i][ind[0], ind[1]] - self.cutoff[i][0]) / (self.cutoff[i][1] - self.cutoff[i][0]))

            self.mask_diffraction_th[i] = self.mask_diffraction[i].to(self.device).float()
            
            # Shifted mask used for the Lofdahl & Scharmer filter
            self.mask_diffraction_shift[i] = np.fft.fftshift(self.mask_diffraction[i].cpu().numpy())

        # Per-patch masks, if several cutoffs have been requested with set_patch_cutoffs()
        if self.patch_cutoff_index is not None:
            self.mask_diffraction_multi = [None] * self.n_o
            for i in range(self.n_o):
                masks = [self._diffraction_mask_from_cutoff(i, c) for c in self.patch_cutoffs[i]]
                self.mask_diffraction_multi[i] = torch.stack(masks, dim=0).to(self.device).float()

        self._setup_radial_bins()

    def _diffraction_mask_from_cutoff(self, i, cutoff):
        """
        Build the diffraction mask of object `i` for a given pair of cutoff frequencies.

        Parameters
        ----------
        i : int
            Index of the object.
        cutoff : list
            Lower and upper frequency of the cosine transition, in units of the
            diffraction cutoff frequency.

        Returns
        -------
        torch.Tensor
            The mask, of shape (n_x, n_y).
        """
        mask = torch.zeros_like(self.rho[i])
        mask[self.rho[i] <= cutoff[0]] = 1.0
        ind = (self.rho[i] > cutoff[0]) & (self.rho[i] <= cutoff[1])
        mask[ind] = torch.cos(np.pi / 2.0 * (self.rho[i][ind] - cutoff[0]) / (cutoff[1] - cutoff[0]))
        return mask

    def set_patch_cutoffs(self, cutoffs, index):
        """
        Use a different diffraction cutoff for different patches.

        A single cutoff has to be a compromise between the parts of the field that carry
        signal up to high frequencies and those that do not. Off-limb patches, for
        instance, have no photospheric signal at all in the wideband channel, so every
        frequency above the noise level only feeds noise into the wavefront estimation.
        This lets each patch use the cutoff that matches its own signal content.

        It has to be called after add_frames() and before deconvolve().

        Parameters
        ----------
        cutoffs : list
            For each object, a list of [lower, upper] cutoff pairs.
        index : list
            For each object, an integer tensor of shape (n_sequences,) selecting which
            pair of `cutoffs` every patch uses.
        """
        self.patch_cutoffs = [list(c) for c in cutoffs]
        self.patch_cutoff_index = [torch.as_tensor(idx).long().cpu() for idx in index]

        # n_o is only known once the frames are combined, so count the objects here
        for i in range(len(self.patch_cutoffs)):
            n_cut = len(self.patch_cutoffs[i])
            if int(self.patch_cutoff_index[i].max()) >= n_cut:
                raise ValueError(f"Cutoff index out of range for object {i}: {int(self.patch_cutoff_index[i].max())} >= {n_cut}")
            counts = torch.bincount(self.patch_cutoff_index[i], minlength=n_cut)
            self.logger.info(f"Object {i} - per-patch cutoffs:")
            for j, c in enumerate(self.patch_cutoffs[i]):
                self.logger.info(f"     - {c} : {int(counts[j])} patches")

    def _diffraction_mask(self, i):
        """
        Diffraction mask of object `i` for the batch of patches being processed, with
        shape (n_batch, n_x, n_y) when per-patch cutoffs are in use and (1, n_x, n_y)
        otherwise, so that it always broadcasts against the batch.
        """
        if self._mask_diff_batch is not None:
            return self._mask_diff_batch[i]
        return self.mask_diffraction_th[i][None, :, :]

    def set_patch_modes(self, n_modes):
        """
        Use a different number of wavefront modes for different patches.

        The wavefront is fitted to each patch independently, so in a patch with little
        signal there is nothing but noise for the modes to fit, and a large basis will
        fit it: the estimate then carries spurious power over the whole band that the
        loss covers, which no noise filter can remove afterwards because it is
        indistinguishable from signal. Off-limb patches are the extreme case. Giving
        them fewer modes removes the freedom to do it, while the patches on the disk
        keep the full basis they can actually constrain.

        It has to be called after add_frames() and before deconvolve().

        Parameters
        ----------
        n_modes : torch.Tensor or array
            Number of modes of every patch, of shape (n_sequences,). Values are clipped
            to the size of the basis.
        """
        self.patch_n_modes = torch.as_tensor(n_modes).long().cpu().clamp(min=2)
        vals, counts = torch.unique(self.patch_n_modes, return_counts=True)
        self.logger.info(f"Per-patch number of modes:")
        for v, c in zip(vals.tolist(), counts.tolist()):
            self.logger.info(f"     - {v} modes : {c} patches")

    def _set_batch_masks(self, seq):
        """
        Select the diffraction mask and the active modes of every patch of the batch.
        """
        if self.patch_cutoff_index is None:
            self._mask_diff_batch = None
        else:
            self._mask_diff_batch = [self.mask_diffraction_multi[i][self.patch_cutoff_index[i][seq].to(self.device)]
                                     for i in range(self.n_o)]

        if self.patch_n_modes is None:
            self._mode_mask_batch = None
        else:
            n = self.patch_n_modes[seq].to(self.device)
            self._mode_mask_batch = (torch.arange(self.n_modes, device=self.device)[None, :] < n[:, None]).float()

    def _setup_radial_bins(self):
        """
        Precompute, for every object, the assignment of each Fourier pixel to a ring of
        constant |nu|. Averaging the filter over these rings brings thousands of samples
        into every estimate instead of one, which is what keeps the filter itself from
        being as noisy as the quantity it is meant to suppress.
        """
        self.rad_index = [None] * self.n_o
        self.rad_center = [None] * self.n_o
        self.rad_count = [None] * self.n_o

        n_bins = self.npix // 2

        for i in range(self.n_o):
            rho = self.rho[i]
            d_rho = float(rho.max()) / n_bins
            idx = torch.clamp((rho / d_rho).long(), max=n_bins - 1).reshape(-1)
            self.rad_index[i] = idx.to(self.device)
            self.rad_count[i] = torch.bincount(idx, minlength=n_bins).to(self.device).float().clamp(min=1.0)
            self.rad_center[i] = (torch.arange(n_bins, device=self.device).float() + 0.5) * d_rho

    def _radial_average(self, x, i):
        """
        Average `x` (n_batch, n_x, n_y) over rings of constant |nu|.

        Returns
        -------
        torch.Tensor
            The radial profiles, of shape (n_batch, n_bins).
        """
        n_batch = x.shape[0]
        n_bins = self.rad_center[i].shape[0]
        out = torch.zeros((n_batch, n_bins), device=x.device, dtype=x.dtype)
        out.index_add_(1, self.rad_index[i], x.reshape(n_batch, -1))
        return out / self.rad_count[i][None, :]

    def wiener_filter(self, Sconj_S, Sconj_I, i):
        """
        Estimate the noise regularization of the object of every patch from the data.

        The object estimated by inverting the multi-frame system,

            O(nu) = sum_j S_j*(nu) D_j(nu) / sum_j |S_j(nu)|^2,

        has a power  |O|^2 = N(nu) / sum_j |S_j|^2 + |O_true|^2,  so the quantity

            Q(nu) = |sum_j S_j* D_j|^2 / sum_j |S_j|^2 = N(nu) + |O_true(nu)|^2 sum_j |S_j(nu)|^2

        splits the power of the estimate into a noise term N, which is the noise power
        spectrum of a single frame and does not depend on the wavefronts, and a signal
        term. Averaging Q over rings of constant |nu| and calibrating N beyond the
        diffraction cutoff - where the telescope cannot transmit signal, so Q is pure
        noise - gives the signal-to-noise ratio of every patch as a function of
        frequency, and with it the Wiener regularization that the object needs.

        Parameters
        ----------
        Sconj_S : torch.Tensor
            sum_j |S_j|^2 for every patch, of shape (n_batch, n_x, n_y).
        Sconj_I : torch.Tensor
            sum_j S_j* D_j for every patch, of shape (n_batch, n_x, n_y).
        i : int
            Index of the object.

        Returns
        -------
        reg : torch.Tensor
            The term to add to sum_j |S_j|^2 in the denominator of the object estimate.
        support : torch.Tensor
            Zero at the frequencies the filter drops entirely.
        weight : torch.Tensor
            The radially averaged Wiener weight, kept for diagnostics.
        """
        centers = self.rad_center[i]
        eps = 1e-20

        Sconj_S = Sconj_S.real
        Q = (torch.conj(Sconj_I) * Sconj_I).real / (Sconj_S + eps)

        Q_r = self._radial_average(Q, i)
        S_r = self._radial_average(Sconj_S, i)

        # Band beyond the diffraction cutoff, where Q measures the noise alone
        lo, hi = self.filter_noise_band[i]
        band = (centers >= lo) & (centers < hi)
        if band.sum() < 2:
            raise ValueError(f"filter_noise_band {self.filter_noise_band[i]} of object {i} contains too few frequencies")

        if self.filter_noise_shape[i] == 'auto':
            # Interpolating the frames (destretching) correlates the noise, so its power
            # spectrum is not flat and a level measured beyond the diffraction cutoff
            # underestimates it at lower frequencies. Recover its shape from the patches
            # themselves: at frequencies above the signal band almost every patch is
            # noise dominated, so a low quantile across patches of the profiles, each
            # normalized by its own level in the calibration band, follows the noise.
            level = torch.median(Q_r[:, band], dim=1).values.clamp(min=eps)
            shape = torch.quantile(Q_r / level[:, None], 0.25, dim=0)

            # Below the signal band the measurement is contaminated by the patches
            # themselves, so hold the shape flat there, and keep it non-increasing:
            # interpolation only ever removes high frequency power.
            shape = torch.cummin(shape, dim=0).values
            n_flat = int((centers < self.filter_noise_flat[i]).sum())
            if 0 < n_flat < centers.shape[0]:
                shape = torch.cat([shape[n_flat].expand(n_flat), shape[n_flat:]])
            shape = shape / torch.median(shape[band])
        else:
            shape = torch.ones_like(centers)

        # Noise level of each patch, from its own profile in the calibration band
        level = torch.median((Q_r / shape[None, :])[:, band], dim=1).values.clamp(min=eps)
        noise_r = level[:, None] * shape[None, :]

        weight = (1.0 - noise_r / Q_r.clamp(min=eps)).clamp(0.0, 1.0)

        # A single ring that happens to fluctuate low must not decide the outcome, so
        # smooth over neighbouring rings first. This matters most for a patch containing
        # the limb: the spectrum of that edge is strongly anisotropic and its radial
        # profile dips, and without smoothing the filter would cut at the dip.
        weight = torch.median(F.pad(weight[:, None, :], (1, 1), mode='replicate')
                              .unfold(-1, 3, 1), dim=-1).values[:, 0, :]

        # A ring can be no better than the best ring inside it: the object's power
        # spectrum falls steeply with frequency, far faster than the noise does, so the
        # signal-to-noise ratio has to be non-increasing. Imposing it drops the rings
        # that are pure noise but fluctuate high, which is what the flood fill of the
        # original filter was for, and unlike a hard support it cannot be tricked into
        # cutting everything away by a single bad ring.
        weight = torch.cummin(weight, dim=1).values

        # Soft threshold. A hard cut leaves a step of `filter_threshold` at the edge of
        # the support, and a step in Fourier space rings in the image - visible as bars
        # along the limb, where the contrast is largest. Rescaling takes the weight to
        # zero continuously instead.
        thr = self.filter_threshold[i]
        weight = ((weight - thr) / max(1.0 - thr, eps)).clamp(0.0, 1.0)

        # Wiener regularization: sum_j |S_j|^2 (1 - w) / w reproduces the weight w where
        # the OTF has its typical power, and additionally damps the individual
        # frequencies at which sum_j |S_j|^2 happens to be close to zero. Where the
        # weight is zero the clamp keeps it finite and `support` removes the frequency.
        reg_r = S_r * (1.0 - weight) / weight.clamp(min=1e-3)

        reg = reg_r[:, self.rad_index[i]].reshape(-1, self.npix, self.npix)
        support = (weight > 0).float()[:, self.rad_index[i]].reshape(-1, self.npix, self.npix)

        return reg, support, weight

    def compute_psfs(self, modes, diversity, jitter=None):
        """
        Compute the Point Spread Functions (PSFs) from the given modes.
        Parameters:
            
            modes (torch.Tensor): A tensor of shape (batch_size, num_modes, height, width) representing the modes.
            diversity (torch.Tensor): A tensor of shape (batch_size, num_diversity, height, width) representing the diversity.
            jitter (torch.Tensor, optional): A tensor representing the jitter. Default is None.

        Returns:
        tuple: A tuple containing:
            - wavefront (torch.Tensor): The computed wavefronts from the estimated modes.
            - psf_norm (torch.Tensor): The normalized PSFs.
            - otf (torch.Tensor): The FFT of the normalized PSFs.
        """
        

        n_seq, n_f, n_active = modes.shape

        # Restrict every patch to the modes it is allowed to use. Zeroing them here also
        # zeroes their gradient, so the ones that are switched off stay at zero.
        if self._mode_mask_batch is not None:
            modes = modes * self._mode_mask_batch[:, None, 0:n_active]
                                        
        psf_norm = [None] * self.n_o
        otf = [None] * self.n_o
        
        if self.use_jitter == 'slow' and n_active == self.n_modes:
                
            # 1. Extract P1 and P2 and reshape for broadcasting
            # Slice shape: [n_seq, nf, 2] -> Unsqueeze to [n_seq, nf, 1, 2]
            P1 = jitter[:, :, 0, :]  # Shape [n_seq, nf, 2]
            P2 = jitter[:, :, 1, :]  # Shape [n_seq, nf, 2]
            
            # Initialize jitter PSF with zeros
            jitter_psf = torch.zeros((n_seq, n_f, self.npix, self.npix), device=self.device)

            for K in range(self.K):

                t = self.tau[K]

                # Compute curve point for step K using the quadratic Bézier formula
                p_k = 2 * (1 - t) * t * P1 + (t ** 2) * P2  # Shape [n_seq, nf, 2]

                # Split into x and y components
                p_x = p_k[..., 0].unsqueeze(-1).unsqueeze(-1)
                p_y = p_k[..., 1].unsqueeze(-1).unsqueeze(-1)

                # 4. Compute 1D separable distances
                # dist_x_sq shape: [n_seq, nf, nx, 1]
                # dist_y_sq shape: [n_seq, nf, 1, ny]
                dist_x_sq = (self.X - p_x) ** 2
                dist_y_sq = (self.Y - p_y) ** 2
                
                # 5. Compute 1D Gaussians
                psf_x = torch.exp(-dist_x_sq / (2 * self.sigma_alias ** 2))
                psf_y = torch.exp(-dist_y_sq / (2 * self.sigma_alias ** 2))

                # 6. Outer product (via broadcasting) creates the [nx, ny] 2D Gaussian
                # Add to the accumulator
                jitter_psf += (psf_x * psf_y)

            jitter_psf /= self.K  # Average over K steps

            # NNormalize to ensure energy conservation
            psf_sum = torch.sum(jitter_psf, dim=(-1, -2), keepdim=True)
            jitter_psf = jitter_psf / (psf_sum + 1e-8)
            jitter_psf = torch.fft.fftshift(jitter_psf, dim=(-2, -1))
            
            # Compute the jitter OTF
            jitter_otf = torch.fft.fft2(jitter_psf)

        if self.use_jitter == 'fast' and n_active > 20:

            sigma_x = torch.exp(jitter[:, :, 0])
            sigma_y = torch.exp(jitter[:, :, 1])
            rho_xy = torch.tanh(jitter[:, :, 2]) * sigma_x * sigma_y

        for i in range(self.n_o):
            # Compute wavefronts from estimated modes                
            wavefront = torch.einsum('ijk,klm->ijlm', modes, self.basis[i][0:n_active, :, :])
            
            # Reuse the same wavefront per object but add the diversity
            wavef = []
            for j in self.init_frame_diversity[i]:                
                div = diversity[i][:, j:j+n_f, None, None] * self.defocus_basis[i][None, None, :, :]
                wavef.append(wavefront + div)
            
            wavef = torch.cat(wavef, dim=1)
            
            # Compute the complex phase
            phase = self.pupil[i][None, None, :, :] * torch.exp(1j * wavef)

            # Compute FFT of the pupil function and compute autocorrelation
            ft = torch.fft.fft2(phase, norm=self.fft_norm)
            psf = (torch.conj(ft) * ft).real
            
            # Normalize PSF        
            psf_norm[i] = psf / torch.sum(psf, [-1, -2], keepdim=True)

            # FFT of the PSF
            otf[i] = torch.fft.fft2(psf_norm[i], norm=self.fft_norm)

            # If using jitter, multiply the OTFs
            if self.use_jitter == 'fast' and n_active > 20:
                # f_x/f_y are defined per object, so the jitter OTF is built here
                tx = self.f_x[i][None, None, :, :]**2 * sigma_x[:, :, None, None]**2
                ty = self.f_y[i][None, None, :, :]**2 * sigma_y[:, :, None, None]**2
                txy = 2 * self.f_x[i][None, None, :, :] * self.f_y[i][None, None, :, :] * rho_xy[:, :, None, None]
                otf[i] = otf[i] * torch.exp(-2 * np.pi**2 * (tx + ty + txy))

            if self.use_jitter == 'slow' and n_active == self.n_modes:
                otf[i] *= jitter_otf
                
        return psf_norm, otf
    
    def compute_psfs_nmf(self, modes, shift_x=None, shift_y=None):
        """
        Compute the Point Spread Functions (PSFs) from the given modes.
        Parameters:
        modes (torch.Tensor): A tensor of shape (batch_size, num_modes, height, width) representing the modes.
        Returns:
        tuple: A tuple containing:
            - wavefront (torch.Tensor): The computed wavefronts from the estimated modes.
            - psf_norm (torch.Tensor): The normalized PSFs.
            - otf (torch.Tensor): The FFT of the normalized PSFs.
        """
                
        n_active = modes.shape[2]
                                        
        psf_norm = [None] * self.n_o
        otf = [None] * self.n_o
        
        # Enforce non-negativity of NMF coefficients with scale from config
        scale = self.config['optimization'].get('softplus_scale', 1.0)
        modes_nn = F.softplus(modes * scale) / scale

        for i in range(self.n_o):

            # Compute PSF from estimated modes                
            psf = torch.einsum('ijk,klm->ijlm', modes_nn, self.basis[i][0:n_active, :, :])

            psf = torch.fft.fftshift(psf, dim=[-2, -1])
                        
            # Normalize PSF (with epsilon to avoid division by zero)        
            psf_norm[i] = psf / (torch.sum(psf, [-1, -2], keepdim=True) + 1e-12)

            # FFT of the PSF
            otf[i] = torch.fft.fft2(psf_norm[i], norm=self.fft_norm)

            # Apply Fourier-domain shift (tip-tilt substitute for NMF)
            # shift_x, shift_y are in pixels; shape: (n_seq,)
            sx = shift_x if shift_x is not None else (self.shift_x if self.config['psf']['shift'] else None)
            sy = shift_y if shift_y is not None else (self.shift_y if self.config['psf']['shift'] else None)
            if sx is not None and sy is not None:
                freq_x = torch.fft.fftfreq(self.npix, device=self.device)  # (W,)
                freq_y = torch.fft.fftfreq(self.npix, device=self.device)  # (H,)
                fy, fx = torch.meshgrid(freq_y, freq_x, indexing='ij')     # (H, W)
                # shift: (n_seq, n_f, 1, 1) for broadcasting over (n_seq, n_f, H, W)
                phase = (-2j * torch.pi * (
                    sx[:, :, None, None] * fx[None, None, :, :] +
                    sy[:, :, None, None] * fy[None, None, :, :]
                ))
                otf[i] = otf[i] * torch.exp(phase)
        
        return psf_norm, otf

    def compute_psfs_vae(self, latent_codes, shift_x=None, shift_y=None):
        """
        Compute the Point Spread Functions (PSFs) from the given latent codes using a VAE model.
        Parameters:
        latent_codes (torch.Tensor): A tensor of shape (batch_size, latent_dim) representing the latent codes.
        Returns:
        tuple: A tuple containing:
            - psf_norm (torch.Tensor): The normalized PSFs.
            - otf (torch.Tensor): The FFT of the normalized PSFs.
        """
        
        n_seq = latent_codes.shape[0]
        
        psf_norm = [None] * self.n_o
        otf = [None] * self.n_o
                
        # Decode latent codes to PSFs using the VAE model
        
        if latent_codes.shape[-1] < 200:
            latent_codes = torch.nn.functional.pad(latent_codes, (0, 200 - latent_codes.shape[-1]))

        latent_codes = rearrange(latent_codes, 'ns nf nz -> (ns nf) nz')  # Reshape for VAE input
        
        decoded_psfs = self.model_vae.decode(latent_codes)  # Shape: (n_seq, 1, H, W)

        decoded_psfs = rearrange(decoded_psfs, '(ns nf) 1 h w -> ns nf h w', ns=n_seq)  # Reshape back to (n_seq, n_f, H, W)

        decoded_psfs = decoded_psfs * (self.x_vae_std + 1e-8) + self.x_vae_mean
        decoded_psfs = torch.exp(decoded_psfs + 1e-10)

        # The VAE generates PSFs of size 64x64. We need to adapt them to the desired patch size
        if decoded_psfs.shape[-1] < self.npix:            
            decoded_psfs = F.pad(decoded_psfs, ((self.npix - decoded_psfs.shape[-2]) // 2, 
                                                (self.npix - decoded_psfs.shape[-2]) // 2, 
                                                (self.npix - decoded_psfs.shape[-1]) // 2, 
                                                (self.npix - decoded_psfs.shape[-1]) // 2))
        if decoded_psfs.shape[-1] > self.npix:
            decoded_psfs = decoded_psfs[:, :, (decoded_psfs.shape[-2] - self.npix) // 2 : (decoded_psfs.shape[-2] + self.npix) // 2,
                                          (decoded_psfs.shape[-1] - self.npix) // 2 : (decoded_psfs.shape[-1] + self.npix) // 2]
            
            

        decoded_psfs = torch.fft.fftshift(decoded_psfs, dim=[-2, -1])
        
        for i in range(self.n_o):
            # Normalize PSF (with epsilon to avoid division by zero)
            psf_norm[i] = decoded_psfs / (torch.sum(decoded_psfs, dim=(-1, -2), keepdim=True) + 1e-12)

            # FFT of the PSF
            otf[i] = torch.fft.fft2(psf_norm[i], norm=self.fft_norm)

            # Apply Fourier-domain shift (tip-tilt substitute for NMF)
            # shift_x, shift_y are in pixels; shape: (n_seq,)                        
            freq_x = torch.fft.fftfreq(self.npix, device=self.device)  # (W,)
            freq_y = torch.fft.fftfreq(self.npix, device=self.device)  # (H,)
            fy, fx = torch.meshgrid(freq_y, freq_x, indexing='ij')     # (H, W)
            # shift: (n_seq, n_f, 1, 1) for broadcasting over (n_seq, n_f, H, W)
            phase = (-2j * torch.pi * (
                shift_x[:, :, None, None] * fx[None, None, :, :] +
                shift_y[:, :, None, None] * fy[None, None, :, :]
            ))
            otf[i] = otf[i] * torch.exp(phase)
        
        return psf_norm, otf
    
    def compute_psf_diffraction(self):
        """
        Compute the Point Spread Functions (PSFs) from diffraction
        
        Returns:
        tuple: A tuple containing:
            - psf_norm (torch.Tensor): The normalized PSFs.
            - otf (torch.Tensor): The FFT of the normalized PSFs.
        """
        
        otf = [None] * self.n_o
        psf_norm = [None] * self.n_o

        for i in range(self.n_o):
            # Compute FFT of the pupil function and compute autocorrelation
            ft = torch.fft.fft2(self.pupil[i], norm=self.fft_norm)
            psf = (torch.conj(ft) * ft).real
            
            # Normalize PSF        
            psf_norm[i] = psf / torch.sum(psf, dim=(-1, -2), keepdim=True)

            # FFT of the PSF
            otf[i] = torch.fft.fft2(psf_norm[i], norm=self.fft_norm)

        return psf_norm, otf
    
    def lofdahl_scharmer_filter(self, Sconj_S, Sconj_I, sigma):
        """
        Applies the Löfdahl-Scharmer filter to the given input tensors.
        Parameters:
        -----------
        Sconj_S : torch.Tensor
            The conjugate of the Fourier transform of the observed image.
        Sconj_I : torch.Tensor
            The conjugate of the Fourier transform of the ideal image.
        Returns:
        --------
        torch.Tensor
            A tensor representing the mask after applying the Löfdahl-Scharmer filter.
        """
        den = torch.conj(Sconj_I) * Sconj_I
        H = (Sconj_S / den).real        
        H = torch.fft.fftshift(H, dim=(-2, -1)).detach().cpu().numpy()
        
        # noise = 1.35 / np.median(H[:, :, 0:10, 0:10], axis=(2,3))

        H = nd.median_filter(H, [1,3,3], mode='wrap')    

        if hasattr(sigma, 'dim') and sigma.dim() > 0:
            if sigma.dim() == 1:
                sigma_1 = sigma[:, None, None].cpu().numpy()
            elif sigma.dim() == 2:
                sigma_1 = sigma[:, :, None].cpu().numpy()
            else:
                sigma_1 = torch.mean(sigma, axis=-1)[:, None, None].cpu().numpy()
        else:
            sigma_1 = sigma.cpu().numpy()
        
        filt = 1.0 - H * sigma_1
        filt[filt < 0.2] = 0.0
        filt[filt > 1.0] = 1.0
                
        nb, nx, ny = filt.shape

        mask = np.zeros_like(filt).astype('float32')

        for ib in range(nb):                
            mask[ib, :, :] = flood(1.0 - filt[ib, :, :], (nx//2, ny//2), tolerance=0.9)
            mask[ib, :, :] = np.fft.fftshift(mask[ib, :, :])
        
        return torch.tensor(mask.astype('float32')).to(Sconj_S.device)

    def get_su_s2(self, obj, sigma, pars_s0=None, pars_s2=None, frames_mean=None):
        """
        Transforms the parameters of the s0 function to ensure they are in the correct range.
        Parameters:
        -----------
        pars_s0 : torch.Tensor
            A tensor containing the parameters for the s0 function.
        i : int
            The index of the object for which to compute s_u.
        Returns:
        --------
        tuple: A tuple containing:
            - K (torch.Tensor): The transformed K parameter.
            - v0 (torch.Tensor): The transformed v0 parameter.
            - p (torch.Tensor): The transformed p parameter.
        """
        if self.loss_type == 'marginal':
            if pars_s2 is not None:
                # Upstream format: pars_s0 shape (n_o, 3), pars_s2 shape (n_o, 1)
                K = torch.exp(pars_s0[obj, 0])            
                v0 = torch.exp(pars_s0[obj, 1])
                p = torch.exp(pars_s0[obj, 2])

                v = self.rho[obj]

                # Evaluate the s_u function on the frequency grid.
                # We use the mean of the frames in Fourier space to set the scale of s_u at frequency 0,0
                s_u = K * self.npix / (1.0 + (v/v0)**2)**(p/2.0)
                
                # The DC component of the Fourier transform of the image is equal to sqrt(nx*ny)*mean if norm='ortho'
                # As a consequence, the variance of the noise at frequency (0,0) is given by s_u[0,0] = mean**2 * nx * ny
                s_u[0, 0] = frames_mean[0]**2 * self.npix * self.npix
                
                s2 = torch.exp(pars_s2[obj, 0])
            else:
                # Local format fallback: pars_s0 shape (batch_size, n_o, 4)
                K = torch.exp(pars_s0[:, obj, 0])[:, None, None]
                v0 = torch.exp(pars_s0[:, obj, 1])[:, None, None]            
                p = torch.exp(pars_s0[:, obj, 2])[:, None, None]

                v = self.rho[obj][None, :, :]

                s_u = K / (1.0 + (v/v0)**2)**(p/2.0)

                s_u[:, 0, 0] = frames_mean**2 * self.npix * self.npix
                
                s2 = torch.exp(pars_s0[:, obj, 3])
        else:
            K, v0, p = None, None, None
            s_u = self.s_u[obj] * torch.ones_like(self.rho[obj]).to(self.device)
            # Average over the frames but keep one noise variance per patch. Collapsing it
            # to a single number makes every patch share the noise of the brightest ones,
            # which for a field crossing the limb is wrong by a large factor.
            if hasattr(sigma[obj], 'dim') and sigma[obj].dim() >= 2:
                s2 = torch.mean(sigma[obj].reshape(sigma[obj].shape[0], -1)**2, dim=1)
            else:
                s2 = torch.mean(sigma[obj]**2)
        
        return s_u, s2, K, v0, p
            
    def solve_object_time_evolution(self, S, D, Reg_Space, lambda_s, lambda_t, contrast=None):
        """
        Solves the batched tridiagonal systems for the multi-burst regularized object using PyTorch.
        
        Parameters:
        -----------
        S : torch.Tensor (complex)
            Shape (M, J, Nx, Ny). The OTFs for M bursts, each containing J frames.
        D : torch.Tensor (complex)
            Shape (M, J, Nx, Ny). The Fourier transformed data frames.
        Reg_Space : torch.Tensor (real)
            Shape (Nx, Ny). Precomputed spatial regularization grid |R_s(u, v)|^2.
        lambda_s : float
            Spatial regularization parameter.
        lambda_t : float
            Temporal regularization parameter.
        device : str
            The target computation device ('cuda' or 'cpu').
            
        Returns:
        --------
        O : torch.Tensor (complex)
            Shape (M, Nx, Ny). The regularized objects for all M bursts on the specified device.
        """
        
        # Detect whether we want a variable lambda_t
        # In such case, weight the lambda_t by the inverse of the contrast of each image in the
        # burst, so that worse images are more regularized than better ones.
        lambda_t_variable = False
        if contrast is not None:            
            contrast_normalized = contrast / torch.min(contrast)
            lambda_t = lambda_t / contrast_normalized**2
            lambda_t = lambda_t[0: -1]
            lambda_t_variable = True
        
        M, J, Nx, Ny = S.shape
        B = Nx * Ny
        
        # 1. Compute Data Components and RHS Vector
        sum_S_sq_flat = torch.sum(torch.abs(S)**2, dim=1).reshape(M, B)
        RHS = torch.sum(torch.conj(S) * D, dim=1).reshape(M, B)
        reg_s_flat = (lambda_s * Reg_Space).flatten()
        
        # 2. Build Full Arrays for Diagonals out-of-place
        # Explicitly tracking lower (a) and upper (c) arrays as requested
        if lambda_t_variable:
            a = -lambda_t.unsqueeze(1).expand(M - 1, B).to(torch.complex128).to(S.device)
            c = -lambda_t.unsqueeze(1).expand(M - 1, B).to(torch.complex128).to(S.device)

                # Build main diagonal list of rows safely
            d_rows = []
            for m in range(M):
                row = sum_S_sq_flat[m, :] + reg_s_flat
                if m == 0:
                    row = row + lambda_t[0]
                elif m == M - 1:
                    row = row + lambda_t[M - 2]
                else:
                    row = row + lambda_t[m - 1] + lambda_t[m]
                d_rows.append(row)

        else:
            a = torch.full((M - 1, B), -lambda_t, dtype=torch.complex128, device=S.device)
            c = torch.full((M - 1, B), -lambda_t, dtype=torch.complex128, device=S.device)
        
            # Build main diagonal list of rows safely
            d_rows = []
            for m in range(M):
                row = sum_S_sq_flat[m, :] + reg_s_flat
                if m == 0 or m == M - 1:
                    row = row + lambda_t
                else:
                    row = row + 2.0 * lambda_t
                d_rows.append(row)

        # 3. Forward Pass using clean out-of-place accumulation lists
        c_prime_list = []
        d_prime_list = []
        
        # Base Row (m = 0)
        d_base = d_rows[0]
        c_prime_list.append(c[0, :] / d_base)
        d_prime_list.append(RHS[0, :] / d_base)
        
        # Intermediate Rows (m = 1 to M-2)
        for m in range(1, M - 1):
            denom = d_rows[m] - a[m - 1, :] * c_prime_list[m - 1]
            c_prime_list.append(c[m, :] / denom)
            d_prime_list.append((RHS[m, :] - a[m - 1, :] * d_prime_list[m - 1]) / denom)
            
        # Last Row (m = M-1)
        if M > 1:
            denom = d_rows[M - 1] - a[M - 2, :] * c_prime_list[M - 2]
            d_prime_list.append((RHS[M - 1, :] - a[M - 2, :] * d_prime_list[M - 2]) / denom)

        # 4. Backward Substitution Pass
        O_list = [None] * M
        O_list[M - 1] = d_prime_list[M - 1]
        
        for m in range(M - 2, -1, -1):
            O_list[m] = d_prime_list[m] - c_prime_list[m] * O_list[m + 1]
            
        # 5. Build Final Tensor Output 
        O_flat = torch.stack(O_list, dim=0)
        return O_flat.reshape(M, Nx, Ny)
    
    def compute_object(self, images_ft, otf, sigma, plane, type_filter='tophat', pars_s0=None, pars_s2=None):
        """
        Compute the object in Fourier space using the specified filter.
        Parameters:
        --------
        images_ft (torch.Tensor): 
            The Fourier transform of the observed images.
        otf (torch.Tensor): 
            The Fourier transform of the point spread function (PSF).
        type_filter (str, optional): 
            The type of filter to use ('tophat'/'scharmer'). Default is 'tophat'.
        Returns:
        --------
        torch.Tensor: The computed object in Fourier space.
        """

        out_ft = [None] * self.n_o
        out_filter_ft = [None] * self.n_o
        out_filter = [None] * self.n_o
        
        for i in range(self.n_o):

            frames_mean = torch.mean(images_ft[i][:, :, 0, 0], dim=1).real

            s_u, s2, K, v0, p = self.get_su_s2(obj=i, sigma=sigma, pars_s0=pars_s0, pars_s2=pars_s2, frames_mean=frames_mean)
            
            # Per-frame noise weighting: γ_kj = mean(σ²) / σ² 
            # We use dimensionless weights to maintain consistency with s2/s_u
            s2_local = torch.mean(sigma[i]**2, dim=1)
            gamma = s2_local[:, None, None, None] / (sigma[i][:, :, None, None] ** 2 + 1e-10)
            Sconj_S = torch.sum(gamma * torch.conj(otf[i]) * otf[i], dim=1)
            Sconj_I = torch.sum(gamma * torch.conj(otf[i]) * images_ft[i], dim=1)
            
            frames_mean = torch.mean(images_ft[i][:, :, 0, 0], dim=1).real

            s_u, s2, K, v0, p = self.get_su_s2(obj=i, sigma=sigma, pars_s0=pars_s0, pars_s2=pars_s2, frames_mean=frames_mean)
            if pars_s2 is not None:
                s2_term = s2
                s2_filt = s2
            else:
                s2_term = s2[:, None, None] if hasattr(s2, 'dim') and s2.dim() > 0 else s2
                s2_filt = s2[:, None] if hasattr(s2, 'dim') and s2.dim() > 0 else s2

            # Use Lofdahl & Scharmer (1994) filter
            if (self.image_filter[i] == 'scharmer'):

                if self.filter_mode[i] == 'wiener':
                    # Weight every frequency by its signal-to-noise ratio instead of
                    # keeping or dropping it, and regularize the inversion with the
                    # noise-to-signal ratio so that the frequencies at which the OTF is
                    # close to a zero are damped as well.
                    reg, mask, self.filter_weight[i] = self.wiener_filter(Sconj_S.detach(), Sconj_I.detach(), i)
                    mask = mask * self._diffraction_mask(i)
                else:
                    reg = None
                    mask = self.lofdahl_scharmer_filter(Sconj_S, Sconj_I, s2_filt.detach()) * self._diffraction_mask(i)
                
                if self.loss_filter == 'wiener_time' or self.loss_filter == 'wiener_time_contrast':
                    Reg_Space = torch.ones_like(Sconj_S[0, ...]).to(self.device)

                    otf_tmp = rearrange(otf[i], '(s t) f w h -> s t f w h', t=self.n_t)
                    images_tmp = rearrange(images_ft[i], '(s t) f w h -> s t f w h', t=self.n_t)
                    n_seq = images_tmp.shape[0]                    
                    out = []
                    for j in range(n_seq):
                        # images = torch.fft.ifft2(images_tmp[j, :, ...]).real
                        # mn = torch.mean(images, dim=(-1, -2, -3))
                        # std = torch.std(images, dim=(-1, -2, -3))
                        # print(mn, std)                        
                        if self.loss_filter == 'wiener_time_contrast':
                            mn_f = torch.mean(images_tmp[j, :, :, 0, 0], dim=-1) / self.npix**2
                            tmp = torch.sum(torch.abs(images_tmp[j, :, ...])**2, dim=(-1, -2)) - images_tmp[j, :, :, 0, 0].real**2
                            tmp = torch.mean(tmp, dim=-1)
                            std_f = torch.sqrt(tmp / self.npix**4)

                            contrast = (std_f / mn_f).real
                        else:
                            contrast = None
                        
                        tmp = self.solve_object_time_evolution(otf_tmp[j, ...], images_tmp[j, ...], Reg_Space, 0.0, self.time_regularization, contrast=contrast)
                        out.append(tmp[None, ...])
                    out = torch.cat(out, dim=0)
                    out = rearrange(out, 's t w h -> (s t) w h')
                    out_ft[i] = out
                    
                    # out_ft[i] = s_u[None, ...] * Sconj_I / (s_u[None, ...] * Sconj_S + s2_term)
                    
                elif reg is not None:
                    out_ft[i] = Sconj_I / (Sconj_S + reg)
                else:
                    out_ft[i] = Sconj_I / (Sconj_S + 1e-10)
                    # out_ft[i] = s_u[None, ...] * Sconj_I / (s_u[None, ...] * Sconj_S + s2_term)
                            
                out_filter_ft[i] = out_ft[i] * mask

                # If we want to enforce positivity of the object, we can use ADMM to solve the optimization problem
                if self.enforce_positive_object:
                    out_filter_ft[i] = self.enforce_positive_admm(Sconj_I, Sconj_S, mask, iters=500)
                
            # Use simple Wiener filter with tophat prior            
            if (self.image_filter[i] == 'tophat'):
                out_ft[i] = Sconj_I / (Sconj_S + s2_term / s_u)
                
                out_filter_ft[i] = out_ft[i] * self._diffraction_mask(i)

            out_filter[i] = torch.fft.ifft2(out_filter_ft[i]).real
            
            # Even though we are enforcing non-negativity, FFT+IFFT can introduce small 
            # negative values due to numerical errors, so we clamp the output to ensure non-negativity.
            if self.enforce_positive_object:
                out_filter[i] = torch.clamp(out_filter[i], min=0.0)
            
            # Add the gradient that we removed
            if self.remove_gradient_apodization:
                 out_filter[i] += plane[i][:, 0, :, :]
        
        return out_ft, out_filter_ft, out_filter
    
    def enforce_positive_admm(self, Sconj_I, Sconj_S, mask, iters=10):
        """
        Enforce positive object using ADMM        

        

        Args:
            Sconj_I (_type_): _description_
            Sconj_S (_type_): _description_
            mask (_type_): _description_
            iters (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """
        O_ft = Sconj_I / (Sconj_S + 1e-10) * mask
        O = torch.fft.ifft2(O_ft).real

        U = torch.zeros_like(O).to(self.device)
        V = O.clone().to(self.device)
        V = torch.clamp(V, min=0.0)

        rho = 1e-4# * Sconj_S.mean().real
        
        t = tqdm(range(iters), desc="ADMM iterations")

        for _ in t:
            # Update O
            O_ft = (Sconj_I + rho * torch.fft.fft2(V - U)) / (Sconj_S + rho) * mask
            O = torch.fft.ifft2(O_ft)                    
            
            # Update V
            V = torch.clamp((O + U).real, min=0.0)

            # Update U
            U = U + O - V

            tmp = OrderedDict()
            tmp['O_min'] = torch.min(O.real).item()
            tmp['O_max'] = torch.max(O.real).item()
            tmp['V_min'] = torch.min(V).item()
            tmp['V_max'] = torch.max(V).item()

            t.set_postfix(ordered_dict=tmp)
                    
        return torch.fft.fft2(V)
    
    def compute_loss(self, frames_ft, otf, sigma, type_filter='tophat', pars_s0=None, pars_s2=None, jitter=None):
        """
        Compute the object in Fourier space using the specified filter.
        Parameters:
        --------
        images_ft (torch.Tensor): 
            The Fourier transform of the observed images.
        otf (torch.Tensor): 
            The Fourier transform of the point spread function (PSF).
        type_filter (str, optional): 
            The type of filter to use ('tophat'/'scharmer'). Default is 'tophat'.
        Returns:
        --------
        torch.Tensor: The computed object in Fourier space.
        """

        out_ft = [None] * self.n_o
        out_filter_ft = [None] * self.n_o
        out_filter = [None] * self.n_o

        loss_data_total = torch.tensor(0.0).to(self.device)
        loss_prior_total = torch.tensor(0.0).to(self.device)
        loss_total = torch.tensor(0.0).to(self.device)

        # Compute the mean of the frames in Fourier space (frequency (0,0) to use it in the s_u prior        
        
        self.n_f = frames_ft[0].shape[1]

        # If we want to do time regularization, first get the object with the Wiener filter
        if self.loss_filter == 'wiener_time' or self.loss_filter == 'wiener_time_contrast':
            obj_ft, obj_filter_ft, obj_filter = self.compute_object(frames_ft, 
                                                                        otf, 
                                                                        sigma, 
                                                                        None, 
                                                                        pars_s0=pars_s0, 
                                                                        pars_s2=pars_s2)
        
        for i in range(self.n_o):            

            if self.loss_filter == 'wiener':

                frames_mean = torch.mean(frames_ft[i][:, :, 0, 0], dim=1).real

                s_u, s2, K, v0, p = self.get_su_s2(obj=i, sigma=sigma, pars_s0=pars_s0, pars_s2=pars_s2, frames_mean=frames_mean)
                                
                # The value of s_u in the case of the joint estimation should be
                # selected by hand to give good results
                # sigma**2/s_u should be the ration between noise and estimated object power spectrum
                # s2 = torch.mean(sigma[i]**2, dim=1) * 10
                
                # m_prior = torch.mean(frames_ft[i],dim=1,keepdims=True)
                # du = frames_ft[i] - m_prior * otf[i]
                
                du = frames_ft[i]
                if pars_s2 is not None:
                    s2_term = s2
                else:
                    s2_term = s2[:, None, None] if hasattr(s2, 'dim') and s2.dim() > 0 else s2

                hu2 = s2_term + s_u * torch.sum(otf[i] * torch.conj(otf[i]), dim=1)
                du2 = torch.sum(du * torch.conj(du), dim=1)
                hu_du = torch.sum(du * torch.conj(otf[i]), dim=1)
                hu_du2 = s_u * hu_du * torch.conj(hu_du)
                                
                loss_data = 0.5 * (du2 - hu_du2 / hu2) / s2_term

                loss_data *= self._diffraction_mask(i)
                
                # If we are doing a marginal estimation of the object, we 
                # # need to add the effect of the marginalized object and also add 
                # a prior on the parameters of s_u and on s2 to keep them in a reasonable range
                if self.loss_type == 'marginal':
                    if pars_s2 is not None:
                        self.pars_s0_avg = [K.detach(), v0.detach(), p.detach(), s2.detach()]
                        self.pars_s0_out[i, 0] = K.detach()
                        self.pars_s0_out[i, 1] = v0.detach()
                        self.pars_s0_out[i, 2] = p.detach()
                        self.pars_s2_out[i, 0] = s2.detach()
                        
                        # Prior loss consequence of the marginalization of the object in the joing loss
                        # plus the term depending on the noise variance
                        loss_prior_marginal = 0.5 * torch.log(hu2) + 0.5 * (self.n_f - 1.0) * torch.log(s2)
                        loss_prior = loss_prior_marginal
                    else:
                        self.pars_s0_avg = [torch.mean(K).detach(), torch.mean(v0).detach(), torch.mean(p).detach(), torch.mean(s2).detach()]
                        self.pars_s0_out[:, i, 0] = K[:, 0, 0].detach()
                        self.pars_s0_out[:, i, 1] = v0[:, 0, 0].detach()
                        self.pars_s0_out[:, i, 2] = p[:, 0, 0].detach()
                        self.pars_s0_out[:, i, 3] = s2.detach()
                        
                        # Prior loss consequence of the marginalization of the object in the joing loss
                        # plus the term depending on the noise variance
                        loss_prior_marginal = 0.5 * torch.log(hu2) + 0.5 * (self.n_f - 1.0) * torch.log(s2[:, None, None])

                        # Prior loss on sigma**2 to avoid zero division and to keep it in a reasonable range
                        # We use a Gaussian prior on log(sigma**2) with mean given by the average of sigma**2 
                        # and a sufficiently large variance to avoid constraining it too much
                        sig2 = torch.mean(sigma[i]**2, dim=1)[:, None, None]
                        loss_prior_s2 = (torch.log(s2[:, None, None]) - torch.log(sig2))**2

                        # Prior loss on K, v0 and p to keep them in a reasonable range

                        # Gaussian prior on log(K) with mean (peak power spectrum for normalized images) and variance 1.0
                        loss_prior_K = 0.5 * (torch.log10(K) - np.log10(self.K_prior[0]))**2 / self.K_prior[1]**2 + torch.log10(K)
                        
                        # Gaussian prior on log(v0) with mean log(0.1) (cutoff frequency for the power spectrum) and variance 1.0
                        loss_prior_v0 = 0.5 * (torch.log10(v0) - np.log10(self.v0_prior[0]))**2 / self.v0_prior[1]**2
                        
                        # Gaussian prior on log(p) with mean log(2.0) (power law index for the power spectrum) and variance 1.0
                        loss_prior_p = 0.5 * (p - self.p_prior[0])**2 / self.p_prior[1]**2

                        loss_prior = loss_prior_marginal + loss_prior_K + loss_prior_p + loss_prior_v0 + loss_prior_s2
                    
                    loss_prior *= self._diffraction_mask(i)

                    loss = loss_data + loss_prior
                else:
                    loss_prior = torch.tensor(0.0).to(self.device)
                    loss = loss_data

            if self.loss_filter == 'wiener_time' or self.loss_filter == 'wiener_time_contrast':

                frames_mean = torch.mean(frames_ft[i][:, :, 0, 0], dim=1).real

                s_u, s2, K, v0, p = self.get_su_s2(obj=i, sigma=sigma, pars_s0=pars_s0, pars_s2=pars_s2, frames_mean=frames_mean)
                                                
                if pars_s2 is not None:
                    s2_term = s2
                else:
                    s2_term = s2[:, None, None] if hasattr(s2, 'dim') and s2.dim() > 0 else s2
                                
                residual = frames_ft[i] - obj_ft[i][:, None, ...] * otf[i]
                loss_data = 0.5 * torch.sum(residual * torch.conj(residual), dim=1) / s2_term
                
                loss_data *= self._diffraction_mask(i)
                                                
                loss_prior = torch.tensor(0.0).to(self.device)
                loss = loss_data
                    
            if self.loss_filter == 'lowpass':

                s_u, s2, K, v0, p = self.get_su_s2(obj=i, sigma=sigma)
                                
                du = frames_ft[i]
                hu2 = s2[:, None, None] + s_u * torch.sum(otf[i] * torch.conj(otf[i]), dim=1)
                du2 = torch.sum(du * torch.conj(du), dim=1)
                hu_du = torch.sum(torch.conj(du) * otf[i], dim=1)
                hu_du2 = s_u * hu_du * torch.conj(hu_du)
                                
                loss_data = 0.5 * (du2 - hu_du2 / hu2) / s2[:, None, None]
                
                # Use Lofdahl & Scharmer (1994) filter
                if (self.image_filter[i] == 'scharmer'):

                    # We assume, for the moment, that the noise is the same for all frames
                    Sconj_S = torch.sum(torch.conj(otf[i]) * otf[i], dim=1)
                    Sconj_I = torch.sum(torch.conj(otf[i]) * frames_ft[i], dim=1)

                    mask = self.lofdahl_scharmer_filter(Sconj_S, Sconj_I, sigma[i]**2) * self._diffraction_mask(i)
                                
                    loss_data *= mask
                    loss_prior = torch.tensor(0.0).to(self.device)
                    loss = loss_data
                
                # Use simple Wiener filter with tophat prior            
                if (self.image_filter[i] == 'tophat'):                    
                    loss = t1 - self._diffraction_mask(i) * t2 * torch.conj(t2) / (Q + 1e-10)

            
            # Weighted version of the loss computation
            # weight_mask = 1.0 + self.rho[i]**2            
            # loss_data_total += torch.mean(weight_mask * loss_data).real
            # loss_prior_total += torch.mean(loss_prior).real
            # loss_total += torch.mean(weight_mask * loss).real

            # Remove zero frequency                     
            loss_data_total += torch.mean(loss_data[:, 1:, 1:].real)
            
            if self.loss_type == 'marginal':
                loss_prior_total += torch.mean(loss_prior[:, 1:, 1:].real)
            loss_total += torch.mean(loss[:, 1:, 1:].real)

            if jitter is not None and self.use_jitter == 'slow':                
                loss_total += 0.01 * torch.mean(jitter[:, :, 0, :]**2) + 0.01 * torch.mean((jitter[:, :, 1, :] - jitter[:, :, 0, :])**2)
        
        return loss_data_total, loss_prior_total, loss_total

    
    def fft_filter(self, image_ft):
        """
        Applies a Fourier filter to the input image in the frequency domain.

        Parameters:
        -----------
        image_ft : torch.Tensor
            The input image in the frequency domain (Fourier transformed).

        Returns:
        --------
        torch.Tensor
            The filtered image in the frequency domain.
        """
        out = [None] * self.n_o
        for i in range(self.n_o):
            out[i] = image_ft[i] * self._diffraction_mask(i)

        return out
            
    def add_frames(self, frames, sigma=None, id_object=0, id_diversity=0, diversity=0.0, XY=None):
        """
        Add frames to the deconvolution object.
        Parameters:
        -----------
        
        frames : torch.Tensor
            The input frames to be deconvolved (n_sequences, n_frames, nx, ny).
        sigma : torch.Tensor
            The noise standard deviation for each object.
        id_object : int, optional
            The object index to which the frames belong (default is 0).
        diversity : torch.Tensor, optional
            The diversity coefficient to use for the deconvolution (n_sequences, n_objects).
            If None, the diversity coefficient is set to zero for all objects.
        Returns:
        --------
        None
        """
        
        self.logger.info(f"Adding frames for object {id_object} - diversity {id_diversity} - defocus {diversity}")

        # If time is included, flatten the frames for sequences and time
        if frames.ndim == 5:                
            self.time_present = True
            self.n_seq, self.n_t, self.n_f, self.nx, self.ny = frames.shape
            frames = rearrange(frames, 's t f h w -> (s t) f h w', t=self.n_t, s=self.n_seq)
            # frames = frames.reshape((-1, self.n_f, self.nx, self.ny))
        else:
            self.time_present = False
            self.n_seq, self.n_f, self.nx, self.ny = frames.shape
            self.n_t = 1
                
        if sigma is None:
            self.logger.info(f"Estimating noise...")            

        if sigma is None:
            self.logger.info(f"Estimating noise...")
            sigma = noise.compute_noise(frames).to(self.device)
            self.logger.info(f"   * Average noise: {torch.mean(sigma)}")            
        else:
            sigma = torch.tensor(sigma).to(self.device)

        self.ind_object.append(id_object)        
        self.ind_diversity.append(id_diversity)

        self.frames.append(frames)
        self.sigma.append(sigma)        

        # If diversity is a scalar, we need to create a tensor of the same size as the number
        # of sequences
        if np.isscalar(diversity):
            diversity = torch.full(frames.shape[0:1], diversity, dtype=torch.float32).to(self.device)
                
        self.diversity.append(diversity)

        if XY is not None:
            if not torch.is_tensor(XY):
                self.XY = XY.astype('float32')
            else:
                self.XY = XY

    def remove_frames(self):
        """
        Add frames to the deconvolution object.
        Parameters:
        -----------
        frames : torch.Tensor
            The input frames to be deconvolved (n_sequences, n_objects, n_frames, nx, ny).
        sigma : torch.Tensor
            The noise standard deviation for each object.
        id_object : int, optional
            The object index to which the frames belong (default is 0).
        diversity : torch.Tensor, optional
            The diversity coefficient to use for the deconvolution (n_sequences, n_objects).
            If None, the diversity coefficient is set to zero for all objects.
        Returns:
        --------
        None
        """
        
        self.logger.info(f"Removing frames for all objects...")

        
        self.ind_object = []
        self.ind_diversity = []

        self.frames = []
        self.sigma = []
        self.diversity = []


    def combine_frames(self):
        """
        Combine the frames from all objects and sequences into a single tensor.
        Observations with different diversity channels are concatenated along the frame axis.
        Returns:
        --------
        torch.Tensor: A tensor of shape (n_sequences, n_objects, n_frames, nx, ny) containing the combined frames.
        """

        self.logger.info(f"Setting up frames...")

        # Get number of objects and number of diversity channels from the added frames
        self.n_bursts = len(self.ind_object)
        self.n_o = max(self.ind_object) + 1

        n_seq, n_f, n_x, n_y = self.frames[0].shape
        
        frames = [None] * self.n_o
        plane = [None] * self.n_o
        diversity = [None] * self.n_o
        sigma = [None] * self.n_o
        index_frames_diversity = [None] * self.n_o

        # Count the number of frames per object, taking into account the diversity channels
        n_frames_per_object = [0] * self.n_o
        n_diversity_per_object = [0] * self.n_o
        for i in range(self.n_bursts):
            n_frames_per_object[self.ind_object[i]] += n_f
            n_diversity_per_object[self.ind_object[i]] += 1
        
        for i in range(self.n_o):
            frames[i] = torch.zeros(n_seq, n_frames_per_object[i], n_x, n_y)
            plane[i] = torch.zeros(n_seq, 1, n_x, n_y)
            diversity[i] = torch.zeros(n_seq, n_frames_per_object[i])
            sigma[i] = torch.zeros(n_seq, n_frames_per_object[i])
            index_frames_diversity[i] = [0] * n_diversity_per_object[i]

        sigma_max = 0.0
        for i in range(self.n_bursts):
            sigma_max = max(sigma_max, torch.max(self.sigma[i]))            

        
        for i in range(self.n_bursts):

            i_obj = self.ind_object[i]
            i_div = self.ind_diversity[i]

            f0 = i_div * n_f
            f1 = (i_div + 1) * n_f

            index_frames_diversity[i_obj][i_div] = f0

            frames[i_obj][:, f0:f1, :, :], subtract = util.apodize(self.frames[i], self.window, gradient=self.remove_gradient_apodization)
            if self.remove_gradient_apodization:
                plane[i_obj][:, :, :, :] = subtract
            
            # Set the diversity for the current object for all frames and for the sequence            
            diversity[i_obj][:, f0:f1] = self.diversity[i][:, None].expand(-1, n_f)
            
            sigma[i_obj][:, f0:f1] = self.sigma[i] #/ sigma_max
                    
        return frames, diversity, index_frames_diversity, sigma, plane
    
    def update_object(self, cutoffs=None):
        """
        Update the object estimate with new cutoffs in the Fourier filter.

        Parameters
        ----------
        cutoffs : list
            A list containing the new cutoffs for each object. Each cutoff contains two numbers, indicating the
            lower and upper frequencies for the transition.
        """

        if self.simultaneous_sequences is None:
            self.logger.info(f"Deconvolution has not been carried out yet")
            return
        
        if cutoffs is None:
            self.logger.info(f"No cutoffs provided")
            return
        
        # Recompute the diffraction masks with the new cutoffs
        self.logger.info('Recalculating object with new cutoffs in the Fourier filter...')

        for i in range(self.n_o):
            self.cutoff[i] = cutoffs[i]
            self.logger.info(f"     - Filter: {self.image_filter[i]} - cutoff: {self.cutoff[i]}...")            
            if self.loss_type == "joint":
                self.logger.info(f"     - s_u: {self.s_u[i]}")
        
        self.compute_diffraction_masks()

        n_seq, _, _, _ = self.frames_apodized[0].shape

        # If time is present, we flatten the sequence and time dimensions to have a single sequence dimension
        # but compute the indices of the new flattened sequences to be able to recover the original shape of 
        # the object at the end of the optimization        
        n_seq = self.n_seq * self.n_t
        ind = np.arange(n_seq)
        n_seq_total = n_seq
                        
        ind = self.split_and_merge(n_seq, self.n_t, self.simultaneous_sequences)

        # # Split sequences in batches
        # ind = np.arange(n_seq)

        # n_seq_total = n_seq

        # # Split the sequences in groups of simultaneous sequences to be computed in parallel
        # ind = np.array_split(ind, np.ceil(n_seq / self.simultaneous_sequences))

        n_sequences = len(ind)
        
        self.psf_seq = [None] * n_sequences        
        self.degraded_seq = [None] * n_sequences
        self.obj_seq = [None] * n_sequences
        self.obj_diffraction_seq = [None] * n_sequences
        
        for i_seq, seq in enumerate(ind):
                            
            if len(seq) > 1:
                self.logger.info(f"Processing sequences [{seq[0]+1},{seq[-1]+1}]/{n_seq_total}")
            else:
                self.logger.info(f"Processing sequence {seq[0]+1}/{n_seq_total}")

            frames_apodized_seq = []
            plane_seq = []
            frames_ft = []
            sigma_seq = []
            diversity_seq = []
            for i in range(self.n_o):
                frames_apodized_seq.append(self.frames_apodized[i][seq, ...].to(self.device))
                plane_seq.append(self.plane[i][seq, ...].to(self.device))
                frames_ft.append(self.frames_ft[i][seq, ...].to(self.device))
                sigma_seq.append(self.sigma[i][seq, ...].to(self.device))
                diversity_seq.append(self.diversity[i][seq, ...].to(self.device))

            self._set_batch_masks(seq)
                
            n_seq = len(seq)

            if self.psf_model.lower() in ['zernike', 'kl']:
                psf, otf = self.compute_psfs(self.modes_seq[i_seq], diversity_seq, jitter=self.jitter_seq[i_seq] if self.use_jitter != 'none' else None)
            
            if self.psf_model.lower() == 'nmf':
                sx, sy = self.shift_seq[i_seq] if self.shift_seq[i_seq] is not None else (None, None)
                psf, otf = self.compute_psfs_nmf(self.modes_seq[i_seq], shift_x=sx, shift_y=sy)
            
            if (self.infer_object):

                # Compute filtered object from the current estimate
                if (self.config['optimization']['transform'] == 'softplus'):
                    obj_ft = torch.fft.fft2(F.softplus(obj), norm=self.fft_norm)
                else:
                    obj_ft = torch.fft.fft2(obj, norm=self.fft_norm)

                # Filter in Fourier
                obj_filter_ft = self.fft_filter(obj_ft)                

            else:
                pars_s0_arg = self.pars_s0_seq[i_seq] if self.loss_type == 'marginal' else None
                pars_s2_arg = self.pars_s2_seq[i_seq] if (self.loss_type == 'marginal' and 'psd' in self.config) else None
                obj_ft, obj_filter_ft, obj_filter = self.compute_object(frames_ft, otf, sigma_seq, plane_seq, pars_s0=pars_s0_arg, pars_s2=pars_s2_arg)  
                                   

            obj_filter_diffraction = [None] * self.n_o
            degraded = [None] * self.n_o
            for i in range(self.n_o):                
                obj_filter_diffraction[i] = torch.fft.ifft2(obj_filter_ft[i] * self.psf_diffraction_ft[i][None, :, :]).real
            
                # Compute final degraded images
                degraded_ft = obj_filter_ft[i][:, None, :, :] * otf[i]
                degraded[i] = torch.fft.ifft2(degraded_ft).real
                        
            for i in range(self.n_o):
                psf[i] = psf[i].detach().cpu()
                degraded[i] = degraded[i].detach().cpu()
                obj_filter[i] = obj_filter[i].detach()
                obj_filter_diffraction[i] = obj_filter_diffraction[i].detach()

            self.psf_seq[i_seq] = psf
            self.degraded_seq[i_seq] = degraded
            self.obj_seq[i_seq] = obj_filter
            self.obj_diffraction_seq[i_seq] = obj_filter_diffraction

            tfinal = time.time()

            # del psf, degraded, obj_filter, obj_filter_diffraction, degraded_ft, obj_ft, obj_filter_ft, otf
        
        # Concatenate the results from all sequences and all objects independently
        # self.psf = [None] * self.n_o
        # self.degraded = [None] * self.n_o
        self.obj = [None] * self.n_o
        self.obj_diffraction = [None] * self.n_o

        # for i in range(self.n_o):
        self.modes = torch.cat(self.modes_seq, dim=0)
                
        for i in range(self.n_o):
            # tmp = [self.psf_seq[j][i] for j in range(n_sequences)]
            # self.psf[i] = torch.cat(tmp, dim=0)

            # tmp = [self.degraded_seq[j][i] for j in range(n_sequences)]
            # self.degraded[i] = torch.cat(tmp, dim=0)

            tmp = [self.obj_seq[j][i] for j in range(n_sequences)]
            self.obj[i] = torch.cat(tmp, dim=0)
            if self.time_present:
                self.obj[i] = rearrange(self.obj[i], '(s t) h w -> s t h w', t=self.n_t, s=self.n_seq)

            tmp = [self.obj_diffraction_seq[j][i] for j in range(n_sequences)]
            self.obj_diffraction[i] = torch.cat(tmp, dim=0)
            if self.time_present:
                self.obj_diffraction[i] = rearrange(self.obj_diffraction[i], '(s t) h w -> s t h w', t=self.n_t, s=self.n_seq)
        
        return 
    
    def write(self, filename, extra=None):
        """
        Write the deconvolved object to a file.
        Parameters:
        -----------
        filename : str
            The name of the file to which the object will be written.
        Returns:
        --------
        None
        """
        
        self.logger.info(f"Writing object to {filename}...")
        
        hdu = [fits.PrimaryHDU(self.modes.cpu().numpy())]
        for i in range(self.n_o):
            tmp = fits.ImageHDU(self.obj[i].cpu().numpy())
            tmp.header['OBJECT'] = f'Object {i+1}'
            if extra is not None:
                for k, v in extra.items():
                    tmp.header[k] = v
            hdu.append(tmp)
        
        hdu = fits.HDUList(hdu)
        hdu.writeto(filename, overwrite=True)

        return

    def split_and_merge(self, M, N, K):
        """
        Split a sequence of length M into chunks of size N, ensuring that the last chunk does not exceed K.
        Parameters:
        -----------
        M : int
            The length of the sequence to be split.
        N : int
            The size of each chunk.
        K : int
            The maximum allowed size for the last chunk.
        Returns:
        --------
        list of lists
            A list containing the split chunks of the sequence.
        """
        if N > K:
            raise ValueError("N cannot be greater than K")
            
        sequence = list(range(M))
        # Calculate the largest multiple of N that is <= K
        step_size = (K // N) * N
        
        # Directly slice the main sequence
        return [sequence[i:i + step_size] for i in range(0, len(sequence), step_size)]
            
    def deconvolve(self,                                    
                   simultaneous_sequences=1,
                   infer_object=False,
                   optimizer='adam',
                   obj_in=None,
                   modes_in=None,
                   n_iterations=20,
                   sequence_length=None):
        

        """
        Perform deconvolution on a set of frames using specified parameters.
        Parameters:
        -----------
        frames : torch.Tensor
            List with the input frames to be deconvolved (n_batch, n_sequences, n_frames, nx, ny) for each object.
        sigma : torch.Tensor
            The noise standard deviation for each object.
        simultaneous_sequences : int, optional
            Number of sequences to be processed simultaneously (default is 1).
        infer_object : bool, optional
            Whether to infer the object during optimization (default is False).
        optimizer : str, optional
            The optimizer to use ('adam' for Adam, 'lbfgs' for LBFGS) (default is 'adam').
        obj_in : torch.Tensor, optional
            Initial object to use for deconvolution (default is None).
        modes_in : torch.Tensor, optional
            Initial modes to use for deconvolution (default is None).
        annealing : bool or str, optional
            Annealing schedule to use ('linear', 'sigmoid', 'none') (default is 'linear'').
        n_iterations : int, optional
            Number of iterations for the optimization (default is 20).        
        Returns:
        --------
        None
        """
                
        # Estimate the modes                
        # modes = self.modalnet(frames)

        self.simultaneous_sequences = simultaneous_sequences
        self.infer_object = infer_object

        _, self.n_f, self.n_x, self.n_y = self.frames[0].shape

        self.logger.info(f" *****************************************")
        self.logger.info(f" *** SPATIALLY INVARIANT DECONVOLUTION ***")
        self.logger.info(f" *****************************************")
        
        # Combine all frames        
        self.frames_apodized, self.diversity, self.init_frame_diversity, self.sigma, self.plane = self.combine_frames()
        
        # Precompute the FFT of the frames to speed up the optimization
        self.frames_ft = [None] * self.n_o
        for i in range(self.n_o):
            self.frames_ft[i] = torch.fft.fft2(self.frames_apodized[i], norm=self.fft_norm)

        # Define all basis
        self._define_basis()
        
        # Fill the list of frames and apodize the frames if needed
        # for i in range(self.n_o):
        #     self.frames_apodized[i] = self.frames_apodized[i].to(self.device)
        #     self.diversity[i] = self.diversity[i].to(self.device)
        #     self.sigma[i] = self.sigma[i].to(self.device)
                    
        self.logger.info(f"Frames")        
        for i in range(self.n_o):
            
            self.logger.info(f"  * Object {i}")            
            self.logger.info(f"     - Number of sequences {self.n_seq}...")
            if self.time_present:
                self.logger.info(f"     - Number of times {self.n_t}...")
            self.logger.info(f"     - Number of frames {self.n_f}...")
            self.logger.info(f"     - Number of diversity channels {len(self.init_frame_diversity[i])}...")
            for j, ind in enumerate(self.init_frame_diversity[i]):
                self.logger.info(f"       -> Diversity {j} = {self.diversity[i][0, ind]} - Noise = {self.sigma[i][0, ind]}...")
            self.logger.info(f"     - Size of frames {self.n_x} x {self.n_y}...")
            self.logger.info(f"     - Filter: {self.image_filter[i]} - cutoff: {self.cutoff[i]}...")
            if self.loss_type == "joint":
                self.logger.info(f"     - s_u: {self.s_u[i]}")
            if self.loss_type == "marginal":                
                self.logger.info(f"     - Stopping optimization after {self.stop_psd} modes")
                                
        self.finite_difference = util.FiniteDifference().to(self.device)
        self.set_regularizations()
                                                    
        # Compute the diffraction masks
        self.compute_diffraction_masks()
        
        # Annealing schedules

        if self.psf_model.lower() in ['zernike', 'kl']:
            modes = np.cumsum(np.arange(2, self.noll_max+1))

        if self.psf_model.lower() == 'nmf':
            n = max(2, (self.n_modes - 2) // 5)
            modes = np.linspace(2, self.n_modes, n).astype(int)

        if self.psf_model.lower() == 'vae':
            n = max(2, (self.n_modes - 2) // 5)
            modes = np.linspace(2, self.n_modes, n).astype(int)
            # modes = (self.n_modes * np.ones(n)).astype(int)
        
        self.anneal = self.compute_annealing(modes, n_iterations)
                
        # If the regularization parameter is a scalar, we assume that it is the same for all objects
        for reg in self.regularization:
            if reg.type == 'iuwt':                
                if not isinstance(reg.lambda_reg, list):
                    reg.lambda_reg = [reg.lambda_reg] * self.n_o

        # Initial values for the 
        if self.loss_type == 'marginal':
            if 'psd' in self.config:
                self.K_prior = [self.config['psd']['K'], 1.0]
                self.v0_prior = [self.config['psd']['v0'], 1.0]
                self.p_prior = [self.config['psd']['p'], 2.0]
                self.logger.info(f"PSD initial values :")
                self.logger.info(f"  - K: {self.K_prior[0]}")
                self.logger.info(f"  - v0: {self.v0_prior[0]}") 
                self.logger.info(f"  - p: {self.p_prior[0]}")
            else:
                self.K_prior = [self.config['priors']['K']['mean'], self.config['priors']['K']['sigma']]
                self.v0_prior = [self.config['priors']['v0']['mean'], self.config['priors']['v0']['sigma']]
                self.p_prior = [self.config['priors']['p']['mean'], self.config['priors']['p']['sigma']]
                self.logger.info(f"Normal hyperpriors parameters :")
                self.logger.info(f"  - K: mean = {self.K_prior[0]}, sigma = {self.K_prior[1]}")
                self.logger.info(f"  - v0: mean = {self.v0_prior[0]}, sigma = {self.v0_prior[1]}") 
                self.logger.info(f"  - p: mean = {self.p_prior[0]}, sigma = {self.p_prior[1]}")

        #--------------------------------
        # Start optimization
        #--------------------------------

        # If time is present, we flatten the sequence and time dimensions to have a single sequence dimension
        # but compute the indices of the new flattened sequences to be able to recover the original shape of 
        # the object at the end of the optimization        
        n_seq = self.n_seq * self.n_t
        ind = np.arange(n_seq)
        n_seq_total = n_seq
                        
        ind = self.split_and_merge(n_seq, self.n_t, simultaneous_sequences)
        
        n_sequences = len(ind)
        
        self.modes_seq = [None] * n_sequences
        self.pars_s0_seq = [None] * n_sequences
        self.pars_s2_seq = [None] * n_sequences
        self.jitter_seq = [None] * n_sequences
        self.shift_seq = [None] * n_sequences
        self.loss = [None] * n_sequences

        self.psf_seq = [None] * n_sequences
        self.degraded_seq = [None] * n_sequences
        self.obj_seq = [None] * n_sequences
        self.obj_diffraction_seq = [None] * n_sequences

        tinit = time.time()
        tinit_global = time.time()

        self.total_time_convergence = 0.0

        if self.psf_model.lower() in ['zernike', 'kl', 'nmf']:
            self.psf_diffraction, self.psf_diffraction_ft = self.compute_psf_diffraction()
        
        for i_seq, seq in enumerate(ind):
            
            if i_seq == 0:
                label_time = ''
            else:
                deltat = tfinal - tinit
                tinit = time.time()
                remaining = deltat * (n_sequences - i_seq)
                label_time = f" - Elapsed time {deltat:.2f} s - Remaining time {remaining:.2f} s ({remaining:.2f} s)"
                
            if len(seq) > 1:
                self.logger.info(f"Processing sequences [{seq[0]+1},{seq[-1]+1}]/{n_seq_total} {label_time}")
            else:
                self.logger.info(f"Processing sequence {seq[0]+1}/{n_seq_total} {label_time}")

            frames_apodized_seq = []
            plane_seq = []
            frames_ft = []
            sigma_seq = []
            diversity_seq = []
            for i in range(self.n_o):
                frames_apodized_seq.append(self.frames_apodized[i][seq, ...].to(self.device))
                plane_seq.append(self.plane[i][seq, ...].to(self.device))
                frames_ft.append(self.frames_ft[i][seq, ...].to(self.device))
                sigma_seq.append(self.sigma[i][seq, ...].to(self.device))
                diversity_seq.append(self.diversity[i][seq, ...].to(self.device))

            self._set_batch_masks(seq)

            n_seq = len(seq)
                                                
            if (infer_object):
                
                obj_init = [None] * self.n_o

                # Find frame with best contrast
                for i in range(self.n_o):
                
                    contrast = torch.std(frames_apodized_seq[i], dim=(-1, -2)) / torch.mean(frames_apodized_seq[i], dim=(-1, -2)) * 100.0
                    ind = torch.argsort(contrast[0, :], descending=True)

                    if obj_in is not None:
                        self.logger.info(f"Using provided initial object...")
                        obj_init[i] = obj_in
                        obj_init[i] = obj_init.to(self.device)
                    else:                    
                        if self.config['initialization']['object'] == 'contrast':
                            self.logger.info(f"Selecting initial object as image with best contrast...")
                            obj_init[i] = frames_apodized_seq[i][:, :, ind[0], :, :]
                        if self.config['initialization']['object'] == 'average':
                            self.logger.info(f"Selecting initial object as average image...")
                            obj_init[i] = torch.mean(frames_apodized_seq[i], dim=2)
                    
                        # Initialize the object with the inverse softplus
                    if (self.config['optimization']['transform'] == 'softplus'):
                        obj_init[i] = torch.log(torch.exp(obj_init[i]) - 1.0)
            
            # Unknown modes
            if modes_in is not None:
                self.logger.info(f"Using provided initial modes...")
                modes = modes_in.clone().detach().to(self.device).requires_grad_(True)
            else:

                # Unknowns for ZERNIKE/KL modes
                if self.psf_model.lower() in ['zernike', 'kl']:
                    if self.config['initialization']['modes_std'] == 0:
                        self.logger.info(f"Initializing modes with zeros...")
                        modes = torch.zeros((n_seq, self.n_f, self.n_modes), device=self.device, requires_grad=True)
                    else:
                        self.logger.info(f"Initializing modes with random values with standard deviation {self.config['initialization']['modes_std']}")
                        modes = self.config['initialization']['modes_std'] * torch.randn((n_seq, self.n_f, self.n_modes))
                        modes = modes.clone().detach().to(self.device).requires_grad_(True)

                # Unknowns for NMF modes
                if self.psf_model.lower() == 'nmf':                               
                    init_mode = self.config.get('initialization', {}).get('modes', 'diffraction')
                    if init_mode == 'cross_spectrum_nnls':
                        self.logger.info("Initializing NMF modes using Cross-Spectrum NNLS...")
                        
                        # Use precomputed uncentered diffraction OTF as baseline reference
                        otf_ref = self.psf_diffraction_ft[0]
                        
                        # Initialize target arrays for the coefficients (shape: n_seq, n_f, n_modes)
                        basis_flat = self.basis[0].reshape((self.n_modes, self.npix**2)) # (n_modes, npix**2)
                        
                        # Collect all target PSFs into a single tensor
                        target_psfs = torch.zeros((n_seq, self.n_f, self.npix, self.npix), device=self.device)
                        
                        # Process per sequence batch and frame to get target PSFs
                        for i_s in range(n_seq):
                            frames_seq = frames_apodized_seq[0][i_s]  # (n_f, npix, npix)
                            frames_ft_seq = torch.fft.fft2(frames_seq, norm=self.fft_norm)  # (n_f, npix, npix)
                            mean_frame_ft = torch.mean(frames_ft_seq, dim=0)  # (npix, npix)
                            
                            # Stabilized division denominator
                            den = torch.conj(mean_frame_ft) * mean_frame_ft
                            mean_den = torch.mean(den).real
                            epsilon = 1e-4 * mean_den + 1e-12
                            
                            for i_f in range(self.n_f):
                                # Ratio of frame OTF to mean frame OTF
                                ratio = (frames_ft_seq[i_f] * torch.conj(mean_frame_ft)) / (den + epsilon)
                                target_otf = ratio * otf_ref
                                target_psf = torch.fft.ifft2(target_otf, norm=self.fft_norm).real
                                target_psfs[i_s, i_f, :, :] = torch.fft.fftshift(target_psf, dim=[-2, -1])
                        
                        # Flatten targets to shape (n_seq * n_f, npix**2)
                        Y = target_psfs.reshape((n_seq * self.n_f, self.npix**2))
                        
                        # Precompute quadratic terms for batched NNLS
                        # H = A * A^T (n_modes, n_modes)
                        H = torch.matmul(basis_flat, basis_flat.t())
                        # G = Y * A^T (n_seq * n_f, n_modes)
                        G = torch.matmul(Y, basis_flat.t())
                        
                        # Compute Lipschitz constant (largest eigenvalue of H) for optimal step size
                        L = torch.linalg.eigvalsh(H)[-1].item()
                        step = 1.0 / L
                        
                        # Batched Accelerated Projected Gradient (APG) for NNLS
                        X = torch.zeros((n_seq * self.n_f, self.n_modes), device=self.device)
                        Y_apg = X.clone()
                        t_apg = 1.0
                        for _ in range(80):
                            # Gradient step & non-negative projection
                            X_next = torch.clamp(Y_apg - step * (torch.matmul(Y_apg, H) - G), min=0.0)
                            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_apg**2))
                            Y_apg = X_next + ((t_apg - 1.0) / t_next) * (X_next - X)
                            X = X_next
                            t_apg = t_next
                            
                        tmp_torch = X.reshape((n_seq, self.n_f, self.n_modes))
                    else:
                        # Centering the diffraction PSF to match the NMF basis spatial layout
                        self.logger.info("Initializing NMF modes using Diffraction PSF NNLS fit...")
                        psf_diff_centered = torch.fft.fftshift(self.psf_diffraction[0], dim=[-2, -1])
                        tmp, _ = optim.nnls(self.basis[0].reshape((self.n_modes, self.npix**2)).T.cpu().numpy(), psf_diff_centered.flatten().cpu().numpy())
                        tmp_torch = torch.tensor(tmp.astype('float32'))[None, None, :].repeat((n_seq, self.n_f, 1))
                    
                    # Convert physical NNLS coefficients to optimization variables
                    # account for softplus_scale: x = inv_softplus(tmp * scale) / scale
                    scale = self.config['optimization'].get('softplus_scale', 1.0)
                    
                    # Safe inverse softplus: for large values x ~= y, for small values use the log formula
                    threshold = 20.0
                    modes_val = torch.where(tmp_torch * scale > threshold, 
                                          tmp_torch, 
                                          torch.log(torch.exp(tmp_torch * scale) - 1.0 + 1e-10) / scale)
                    
                    modes = modes_val.clone().detach().to(self.device).requires_grad_(True)
                    
                    # Initialize learnable PSF shift (one per sequence, in pixels) if enabled in config
                    if self.config['psf']['shift']:
                        self.shift_x = torch.zeros((n_seq, self.n_f), device=self.device, requires_grad=True)
                        self.shift_y = torch.zeros((n_seq, self.n_f), device=self.device, requires_grad=True)

                # Unknowns for VAE modes
                if self.psf_model.lower() == 'vae':
                    self.logger.info(f"Initializing VAE modes with random values...")
                    modes = torch.randn((n_seq, self.n_f, self.n_modes), device=self.device, requires_grad=True)
                    shift_x = torch.zeros((n_seq, self.n_f), device=self.device, requires_grad=True)
                    shift_y = torch.zeros((n_seq, self.n_f), device=self.device, requires_grad=True)
                    
            if self.psf_model.lower() == 'nmf' and self.config['psf']['shift']:
                self.logger.info(f"NMF: adding learnable PSF shift (shift_x, shift_y) to optimizer...")

            if (infer_object):
                self.logger.info(f"Optimizing object and modes...")

                parameters = [{'params': modes, 'lr': self.lr_modes}]
                if self.psf_model.lower() == 'nmf' and self.config['psf']['shift']:
                    parameters.append({'params': self.shift_x, 'lr': self.lr_modes})
                    parameters.append({'params': self.shift_y, 'lr': self.lr_modes})
                obj = [None] * self.n_o

                for i in range(self.n_o):
                    obj[i] = obj_init[i].clone().detach().to(self.device).requires_grad_(True)
                    parameters.append({'params': obj[i], 'lr': self.lr_obj})
                                    
            else:
                self.logger.info(f"Optimizing modes only...")

                parameters = [{'params': modes, 'lr': self.lr_modes}]
                if self.psf_model.lower() == 'nmf' and self.config['psf']['shift']:
                    parameters.append({'params': self.shift_x, 'lr': self.lr_modes})
                    parameters.append({'params': self.shift_y, 'lr': self.lr_modes})

                if self.psf_model.lower() == 'vae' and self.config['psf']['shift']:
                    parameters.append({'params': shift_x, 'lr': self.lr_modes})
                    parameters.append({'params': shift_y, 'lr': self.lr_modes})

                if self.loss_type == 'marginal':
                    if 'psd' in self.config:
                        # Upstream format: pars_s0 shape (n_o, 3), pars_s2 shape (n_o, 1)
                        pars_s0 = np.zeros((self.n_o, 3))
                        for i in range(self.n_o):
                            pars_s0[i, 0] = np.log(self.K_prior[0] / self.npix)
                            pars_s0[i, 1] = np.log(self.v0_prior[0])
                            pars_s0[i, 2] = np.log(self.p_prior[0])
                        self.pars_s0_out = np.zeros((self.n_o, 3))
                        pars_s0_torch = torch.tensor(pars_s0.astype('float32')).to(self.device).requires_grad_(True)
                        self.pars_s0_out = torch.tensor(self.pars_s0_out.astype('float32')).to(self.device)
                        if self.lr_prior > 0:
                            parameters.append({'params': pars_s0_torch, 'lr': self.lr_prior})

                        pars_s2 = np.zeros((self.n_o, 1))
                        for i in range(self.n_o):                        
                            pars_s2[i, 0] = np.log(sigma_seq[i].mean().item()**2)
                        self.pars_s2_out = np.zeros((self.n_o, 1))
                        pars_s2_torch = torch.tensor(pars_s2.astype('float32')).to(self.device).requires_grad_(True)
                        self.pars_s2_out = torch.tensor(self.pars_s2_out.astype('float32')).to(self.device)
                        if self.lr_prior > 0:
                            parameters.append({'params': pars_s2_torch, 'lr': self.lr_prior})
                    else:
                        # Local format fallback: pars_s0 shape (n_seq, n_o, 4)
                        pars_s0 = np.zeros((n_seq, self.n_o, 4))
                        for i in range(self.n_o):
                            pars_s0[:, i, 0] = np.log(self.K_prior[0])
                            pars_s0[:, i, 1] = np.log(self.v0_prior[0])
                            pars_s0[:, i, 2] = self.p_prior[0]
                            pars_s0[:, i, 3] = np.log(sigma_seq[i].mean().item()**2)
                        self.pars_s0_out = np.zeros((n_seq, self.n_o, 4))
                        pars_s0_torch = torch.tensor(pars_s0.astype('float32')).to(self.device).requires_grad_(True)
                        self.pars_s0_out = torch.tensor(self.pars_s0_out.astype('float32')).to(self.device)
                        pars_s2_torch = None
                        if self.lr_prior > 0:
                            parameters.append({'params': pars_s0_torch, 'lr': self.lr_prior})
                    for i in range(self.n_o):                        
                        pars_s2[i, 0] = np.log(sigma_seq[i].mean().item()**2)
                    self.pars_s2_out = np.zeros((self.n_o, 1))
                    pars_s2_torch = torch.tensor(pars_s2.astype('float32')).to(self.device).requires_grad_(True)
                    self.pars_s2_out = torch.tensor(self.pars_s2_out.astype('float32')).to(self.device)
                    parameters.append({'params': pars_s2_torch, 'lr': self.lr_prior})

                jitter_torch = None
                # Fast jitter, faster than the integration time
                if self.use_jitter == 'fast':
                    jitter = np.zeros((n_seq, self.n_f, 3))
                    jitter[:, :, 0] = -2.0
                    jitter[:, :, 1] = -2.0
                    jitter[:, :, 2] = 0.01

                    jitter_torch = torch.tensor(jitter.astype('float32')).to(self.device).requires_grad_(True)
                    if self.lr_modes > 0:
                        parameters.append({'params': jitter_torch, 'lr': self.lr_modes})

                # Slow jitter, parameterized in terms of a Bezier curve with 3 ccontrol points
                # The first one is always at (0, 0)
                if self.use_jitter == 'slow':
                    jitter = 0.3*np.random.randn(n_seq, self.n_f, 2, 2)
                    jitter = np.zeros((n_seq, self.n_f, 2, 2))
                    
                    jitter_torch = torch.tensor(jitter.astype('float32')).to(self.device).requires_grad_(True)
                    parameters.append({'params': jitter_torch, 'lr': self.lr_modes})

            # Second order optimizer
            if optimizer == 'lbfgs':
                self.logger.info(f"Using LBFGS optimizer...")
                # LBFGS does not support parameter groups (different LRs for different params)
                # We flatten the parameters into a single list
                params_lbfgs = []
                for group in parameters:
                    if isinstance(group, dict):
                        params_lbfgs.append(group['params'])
                    else:
                        params_lbfgs.append(group)
                opt = torch.optim.LBFGS(params_lbfgs, lr=self.lr_modes, line_search_fn='strong_wolfe')
            if optimizer == 'adam':
                self.logger.info(f"Using Adam optimizer...")
                opt = torch.optim.Adam(parameters)
            if optimizer == 'adamw':
                self.logger.info(f"Using AdamW optimizer...")
                opt = torch.optim.AdamW(parameters)
            if optimizer == 'cg':
                if NGC_OPTIMIZER:
                    self.logger.info(f"Using CG optimizer...")
                    opt = ncg_optimizer.BASIC(parameters, method = 'CD', line_search = 'Strong_Wolfe', c1 = 1e-4, c2 = 0.9, lr = 1, rho = 0.5, eps=1e-8)
                else:
                    self.logger.info(f"Using Adam optimizer...")
                    opt = torch.optim.Adam(parameters)

            if optimizer not in ['lbfgs', 'adam', 'adamw', 'cg']:
                raise ValueError(f"Optimizer {optimizer} not supported")
            
            self.logger.info(f"Starting optimization with {n_iterations} iterations...")
            self.logger.info(f"Learning rates: modes {self.lr_modes} - object {self.lr_obj if infer_object else 'N/A'} - psd prior {self.lr_prior if not infer_object and self.loss_type == 'marginal' else 'N/A'}")
                
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3*n_iterations)

            losses = torch.zeros(n_iterations, device=self.device)

            t = tqdm(range(n_iterations))

            if self.psf_model.lower() in ['zernike', 'kl']:
                n_active = 2
            
            if self.psf_model.lower() == 'nmf':
                n_active = self.n_modes

            if self.psf_model.lower() == 'vae':
                n_active = self.n_modes

            modes_previous = modes.clone().detach()

            self.t0_convergence = time.time()

            flip = 0
                
            for loop in t:
                                
                def closure():
                            
                    opt.zero_grad(set_to_none=True)

                    if self.psf_model.lower() in ['zernike', 'kl']:
                
                        # Compute PSF from current wavefront coefficients and reference 
                        modes_centered = modes.clone()
                        modes_centered[:, : , 0:2] = modes_centered[:, :, 0:2] - modes[:, 0:1, 0:2]

                        if self.remove_tt:
                            modes_centered[:, :, 0:2] = 0.0
                                    
                        # modes -> (n_seq, n_f, self.n_modes)
                        # jitter > (n_seq, n_f, 3)                                                
                        psf, otf = self.compute_psfs(modes_centered[..., 0:n_active], diversity_seq, jitter=jitter_torch if self.use_jitter != 'none' else None)
                    
                    if self.psf_model.lower() == 'nmf':
                        psf, otf = self.compute_psfs_nmf(modes[..., 0:n_active])

                    if self.psf_model.lower() == 'vae':

                        # Remove the tip-tilt of the first frame of each sequence to avoid degeneracy in the optimization
                        shift_x_centered = shift_x.clone()
                        shift_y_centered = shift_y.clone()
                        shift_x_centered = shift_x_centered - shift_x[:, 0:1]
                        shift_y_centered = shift_y_centered - shift_y[:, 0:1]
                        psf, otf = self.compute_psfs_vae(modes[..., 0:n_active], shift_x=shift_x_centered, shift_y=shift_y_centered)
                    
                    obj_filter = None
                    if (infer_object):

                        obj_ft = [None] * self.n_o
                        obj_filter_ft = [None] * self.n_o

                        loss_mse = torch.tensor(0.0).to(self.device)
                        
                        for i in range(self.n_o):
                            # Compute filtered object from the current estimate while also clamping negative values
                            if (self.config['optimization']['transform'] == 'softplus'):
                                tmp = torch.clamp(F.softplus(obj[i]), min=0.0)
                                obj_ft[i] = torch.fft.fft2(tmp, norm=self.fft_norm)
                            else:
                                tmp = torch.clamp(obj[i], min=0.0)
                                obj_ft[i] = torch.fft.fft2(tmp, norm=self.fft_norm)
                        
                        # Filter in Fourier
                        obj_filter_ft = self.fft_filter(obj_ft)

                        for i in range(self.n_o):

                            degraded_ft = obj_ft[i][:, None, :, :] * otf[i]

                            # residual = self.weight[:, :, None, None, None] * (degraded_ft - frames_ft)
                            residual =  (degraded_ft - frames_ft[i])
                            loss_mse += torch.mean((residual * torch.conj(residual)).real) / self.npix**2

                    else:                        

                        pars_s0_arg = pars_s0_torch if self.loss_type == 'marginal' else None
                        pars_s2_arg = pars_s2_torch if (self.loss_type == 'marginal' and 'psd' in self.config) else None

                        if self.show_object_info:
                            obj_ft, obj_filter_ft, obj_filter = self.compute_object(frames_ft, 
                                                                                    otf, 
                                                                                    sigma_seq, 
                                                                                    plane_seq, 
                                                                                    pars_s0=pars_s0_arg, 
                                                                                    pars_s2=pars_s2_arg)

                        # Compute the loss function using the appropriate filter for the loss
                        loss_data, loss_prior, loss = self.compute_loss(frames_ft, 
                                                                        otf, 
                                                                        sigma_seq, 
                                                                        pars_s0=pars_s0_arg, 
                                                                        pars_s2=pars_s2_arg,
                                                                        jitter=jitter_torch if self.use_jitter != 'none' else None)

                    # If MOMFBD is used, then the object cannot be regularized. Look for alternatives for the future
                    # Object regularization
                    loss_obj = torch.tensor(0.0).to(self.device)
                    if obj_filter is not None:
                        for index in self.index_regularization['object']:
                            loss_obj += self.regularization[index](obj_filter)
                    
                    # Total loss
                    loss += loss_obj
                                                        
                    # Save some information for the progress bar
                    # self.loss_local = loss.detach()
                    # self.obj_filter = [None] * self.n_o
                    # for i in range(self.n_o):
                    #     self.obj_filter[i] = obj_filter[i].detach()
                    # self.loss_mse_local = loss_mse.detach()
                    # self.loss_obj_local = loss_obj.detach()

                    if optimizer == 'cg' and NGC_OPTIMIZER:
                        loss.backward(retain_graph=True)
                    else:
                        loss.backward()

                    self.loss_data_local = loss_data.detach()
                    self.loss_prior_local = loss_prior.detach()
                    self.loss_local = loss.detach()
                    self.loss_obj_local = loss_obj.detach()

                    if self.show_object_info:
                        self.obj_filter_local = obj_filter
                    
                    return loss
                        
                loss = opt.step(closure)

                losses[loop] = loss.detach()

                loss_data = self.loss_data_local
                loss_prior = self.loss_prior_local
                loss = self.loss_local
                loss_obj = self.loss_obj_local
                
                if self.show_object_info:
                    obj_filter = self.obj_filter_local

                # Reduce the learning rate of the PSD parameters
                # if self.loss_type == 'marginal':
                    # opt.param_groups[1]['lr'] *= 0.99                                
                
                if self.handle is not None:
                    gpu_usage = f'{self.handle.gpu_utilization():03d}'
                    memory_usage = f'{self.handle.memory_used() / 1024**2:4.1f}/{self.handle.memory_total() / 1024**2:4.1f} MB'
                    memory_pct = f'{self.handle.memory_used() / self.handle.memory_total() * 100.0:4.1f}%'
                                   
                tmp = OrderedDict()
                
                if self.cuda:
                    tmp['gpu'] = f'{gpu_usage} %'
                    tmp['mem'] = f'{memory_usage}'

                delta_modes = torch.mean(torch.abs(modes.detach() - modes_previous))

                modes_previous = modes.clone().detach()

                tmp['active'] = f'{n_active}'
                if self.show_object_info:
                    tmp['contrast'] = f'{torch.std(obj_filter[0]) / torch.mean(obj_filter[0]) * 100.0:7.4f}'
                    tmp['minmax'] = f'{torch.min(obj_filter[0]):7.4f}/{torch.max(obj_filter[0]):7.4f}'
                tmp['chg'] = f'{delta_modes.item():8.6f}'
                if self.loss_type == 'marginal' and len(opt.param_groups) > 1:
                    tmp['lr'] = f'{opt.param_groups[0]["lr"]:6.3f}/{opt.param_groups[1]["lr"]:6.3f}'
                else:
                    tmp['lr'] = f'{opt.param_groups[0]["lr"]:6.3f}'
                tmp['LDATA'] = f'{loss_data.detach().item():8.6f}'
                tmp['LPRIOR'] = f'{loss_prior.detach().item():8.6f}'
                if self.loss_type == 'marginal':
                    tmp['K'] = f'{self.pars_s0_out[0][0].item() * self.npix:6.2f}'
                    tmp['v0'] = f'{self.pars_s0_out[0][1].item():6.2f}'
                    tmp['p'] = f'{self.pars_s0_out[0][2].item():6.2f}'
                    tmp['sig'] = f'{np.sqrt(self.pars_s2_out[0][0].item()):7.4f}'
                if self.use_jitter == 'fast':
                    tmp['sx'] = f'{torch.min(torch.exp(jitter_torch[:, :, 0])).item():6.3f}'
                    tmp['sy'] = f'{torch.min(torch.exp(jitter_torch[:, :, 1])).item():6.3f}'
                    tmp['rxy'] = f'{torch.min(torch.tanh(jitter_torch[:, :, 2])).item():6.3f}'
                if self.use_jitter == 'slow':
                    tmp['px'] = f'{torch.min((jitter_torch[:, :, :, 0])).item():6.3f}/{torch.max((jitter_torch[:, :, :, 0])).item():6.3f}'
                    tmp['py'] = f'{torch.min((jitter_torch[:, :, :, 1])).item():6.3f}/{torch.max((jitter_torch[:, :, :, 1])).item():6.3f}'
                tmp['L'] = f'{loss.detach().item():8.6f}'
                t.set_postfix(ordered_dict=tmp)
                
                n_active = self.anneal[loop]
                if self.loss_type == 'marginal':
                    if n_active > self.stop_psd:
                        if len(opt.param_groups) > 1:
                            opt.param_groups[1]['lr'] = 0.0

                
            self.tf_convergence = time.time()

            self.total_time_convergence += self.tf_convergence - self.t0_convergence
            
            if self.psf_model.lower() in ['zernike', 'kl']:
                modes_centered = modes.clone().detach()
                modes_centered[:, :, 0:2] = modes_centered[:, :, 0:2] - modes_centered[:, 0:1, 0:2]
                psf, otf = self.compute_psfs(modes_centered, diversity_seq, jitter=jitter_torch if self.use_jitter != 'none' else None)
            
            if self.psf_model.lower() == 'nmf':
                psf, otf = self.compute_psfs_nmf(modes)

            if self.psf_model.lower() == 'vae':
                shift_x_centered = shift_x.clone()
                shift_y_centered = shift_y.clone()
                shift_x_centered = shift_x_centered - shift_x[:, 0:1]
                shift_y_centered = shift_y_centered - shift_y[:, 0:1]
                psf, otf = self.compute_psfs_vae(modes, shift_x=shift_x_centered, shift_y=shift_y_centered)
                
            if (infer_object):
                
                # Compute filtered object from the current estimate
                obj_ft = [None] * self.n_o
                obj_filter_ft = [None] * self.n_o
                obj_filter = [None] * self.n_o
                
                for i in range(self.n_o):
                    # Compute filtered object from the current estimate while also clamping negative values
                    if (self.config['optimization']['transform'] == 'softplus'):
                        scale = self.config['optimization'].get('softplus_scale', 1.0)
                        tmp = torch.clamp(F.softplus(obj[i] * scale) / scale, min=0.0)
                        obj_ft[i] = torch.fft.fft2(tmp, norm=self.fft_norm)
                    else:
                        tmp = torch.clamp(obj[i], min=0.0)
                        obj_ft[i] = torch.fft.fft2(tmp, norm=self.fft_norm)
                
                # Filter in Fourier
                obj_filter_ft = self.fft_filter(obj_ft)                

                for i in range(self.n_o):
                    obj_filter[i] = torch.fft.ifft2(obj_filter_ft[i]).real

            else:
                pars_s0_arg = pars_s0_torch if self.loss_type == 'marginal' else None
                pars_s2_arg = pars_s2_torch if (self.loss_type == 'marginal' and 'psd' in self.config) else None
                obj_ft, obj_filter_ft, obj_filter = self.compute_object(frames_ft, otf, sigma_seq, plane_seq, pars_s0=pars_s0_arg, pars_s2=pars_s2_arg)
                                   

            obj_filter_diffraction = [None] * self.n_o
            degraded = [None] * self.n_o
            for i in range(self.n_o):                
                if self.psf_model.lower() in ['zernike', 'kl']:
                    obj_filter_diffraction[i] = torch.fft.ifft2(obj_filter_ft[i] * self.psf_diffraction_ft[i][None, :, :]).real
            
                # Compute final degraded images
                degraded_ft = obj_filter_ft[i][:, None, :, :] * otf[i]
                degraded[i] = torch.fft.ifft2(degraded_ft).real
            
            # Store the results for the current set of sequences
            self.modes_seq[i_seq] = modes.detach()
            self.jitter_seq[i_seq] = jitter_torch.detach() if (self.use_jitter != 'none' and jitter_torch is not None) else None
            if self.psf_model.lower() == 'nmf' and self.config['psf']['shift']:
                self.shift_seq[i_seq] = (self.shift_x.detach(), self.shift_y.detach())
            self.pars_s0_seq[i_seq] = self.pars_s0_out if not infer_object and self.loss_type == 'marginal' else None
            self.pars_s2_seq[i_seq] = self.pars_s2_out if not infer_object and self.loss_type == 'marginal' and 'psd' in self.config else None
            self.loss[i_seq] = losses.detach()

            for i in range(self.n_o):
                psf[i] = psf[i].detach().cpu()
                degraded[i] = degraded[i].detach().cpu()
                obj_filter[i] = obj_filter[i].detach()
                if self.psf_model.lower() in ['zernike', 'kl']:
                    obj_filter_diffraction[i] = obj_filter_diffraction[i].detach()

            self.psf_seq[i_seq] = psf
            self.degraded_seq[i_seq] = degraded
            self.obj_seq[i_seq] = obj_filter
            self.obj_diffraction_seq[i_seq] = obj_filter_diffraction

            tfinal = time.time()

            # del psf, degraded, obj_filter, obj_filter_diffraction, degraded_ft, obj_ft, obj_filter_ft, otf
        
        deltat = tfinal - tinit
        deltat_global = tfinal - tinit_global        
        self.total_time = deltat_global        
        self.logger.info(f"Elapsed time {deltat:.2f} s - Total time: {deltat_global:.2f} s")

        # Concatenate the results from all sequences and all objects independently
        # self.psf = [None] * self.n_o
        # self.degraded = [None] * self.n_o
        self.obj = [None] * self.n_o
        self.obj_diffraction = [None] * self.n_o
        self.pars_s0 = [None] * self.n_o        

        # for i in range(self.n_o):
        self.modes = torch.cat(self.modes_seq, dim=0)
        self.loss = torch.cat(self.loss, dim=0)
        self.jitter = torch.cat(self.jitter_seq, dim=0) if (self.use_jitter != 'none' and self.jitter_seq[0] is not None) else None
                
        for i in range(self.n_o):
            # tmp = [self.psf_seq[j][i] for j in range(n_sequences)]
            # self.psf[i] = torch.cat(tmp, dim=0)

            # tmp = [self.degraded_seq[j][i] for j in range(n_sequences)]
            # self.degraded[i] = torch.cat(tmp, dim=0)
            
            tmp = [self.obj_seq[j][i] for j in range(n_sequences)]            
            self.obj[i] = torch.cat(tmp, dim=0)

            # Put back time dimension if it was present in the input data
            if self.time_present:
                self.obj[i] = rearrange(self.obj[i], '(s t) h w -> s t h w', t=self.n_t, s=self.n_seq)

            tmp = [self.obj_diffraction_seq[j][i] for j in range(n_sequences)]

            if self.psf_model.lower() in ['zernike', 'kl']:
                self.obj_diffraction[i] = torch.cat(tmp, dim=0)
                if self.time_present:
                    self.obj_diffraction[i] = rearrange(self.obj_diffraction[i], '(s t) h w -> s t h w', t=self.n_t, s=self.n_seq)
            
            if self.loss_type == 'marginal':
                if 'psd' in self.config:
                    tmp_s0 = [self.pars_s0_seq[j][i, :] for j in range(n_sequences)]
                    tmp_s2 = [self.pars_s2_seq[j][i, :] for j in range(n_sequences)]
                    if len(tmp_s0) > 0:
                        self.pars_s0[i] = torch.stack(tmp_s0, dim=0)
                        # We can also store pars_s2 as an attribute if needed
                        self.pars_s2 = [None] * self.n_o
                        self.pars_s2[i] = torch.stack(tmp_s2, dim=0)
                    else:
                        self.pars_s0[i] = None
                else:
                    tmp = [self.pars_s0_seq[j][:, i, :] for j in range(n_sequences)]
                    if len(tmp) > 0:
                        self.pars_s0[i] = torch.cat(tmp, dim=0)
                    else:
                        self.pars_s0[i] = None
                    
        return 
    
    
if __name__ == '__main__': 
    pass
