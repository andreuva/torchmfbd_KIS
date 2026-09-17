def _check_config(config):

    # Telescope
    if "diameter" not in config["telescope"]:
        raise ValueError("diameter is mandatory")
    
    if "central_obscuration" not in config["telescope"]:
        config["telescope"]["central_obscuration"] = 0.0

    if "spider" not in config["telescope"]:
        config["telescope"]["spider"] = 0

    # Images
    if "n_pixel" not in config["images"]:
        raise ValueError("n_pixel is mandatory")
    
    if "pix_size" not in config["images"]:
        raise ValueError("pix_size is mandatory")
    
    if "apodization_border" not in config["images"]:
        config["images"]["apodization_border"] = 0
    
    if "remove_gradient_apodization" not in config["images"]:
        config["images"]["remove_gradient_apodization"] = False

    # Optimization
    if "gpu" not in config["optimization"]:
        config["optimization"]["gpu"] = -1

    if "transform" not in config["optimization"]:
        config["optimization"]["transform"] = "none"

    if "softplus_scale" not in config["optimization"]:
        config["optimization"]["softplus_scale"] = 1.0
    
    if "lr_obj" not in config["optimization"]:
        config["optimization"]["lr_obj"] = 0.01

    if "lr_modes" not in config["optimization"]:
        config["optimization"]["lr_modes"] = 0.01
    
    if "lr_prior" not in config["optimization"]:
        config["optimization"]["lr_prior"] = 0.02

    if "jitter" not in config["psf"]:
        config["psf"]["jitter"] = 'none'

    if "enforce_positive_object" not in config["optimization"]:
        config["optimization"]["enforce_positive_object"] = False
        
    if "shift" not in config["psf"]:
        config["psf"]["shift"] = False

    if "remove_tt" not in config["psf"]:
        config["psf"]["remove_tt"] = False

    if config["optimization"]["transform"] not in ["softplus", "none"]:
        raise ValueError(f"Invalid value for transform. It is {config['optimization']['transform']} but should be softplus or none")
    
    if config["initialization"]["object"] not in ["contrast", "average"]:
        raise ValueError(f"Invalid value for initialization of the object. It is {config['initialization']['object']} but should be contrast or average")
    
    if config["annealing"]["type"] not in ["sigmoid", "linear", "none"]:
        raise ValueError(f"Invalid value for annealing type. It is {config['annealing']['type']} but should be sigmoid, linear or none")
        
    if config["psf"]["model"] not in ["zernike", "kl", "pca", "nmf", "vae"]:
        raise ValueError(f"Invalid value for psf model. It is {config['psf']['model']} but should be zernike, kl, pca or nmf")
    
    for k, v in config.items():
        if "object" in k:
            if v["image_filter"] not in ["tophat", "scharmer", "none"]:
                raise ValueError(f"Invalid value for image_filter. It is {v['image_filter']} but should be tophat, scharmer or none")
            if "s_u_joint" not in v:
                v["s_u_joint"] = 100.0

            # How the Loefdahl & Scharmer noise filter is turned into a Fourier filter.
            #   'mask'   : the original binary flood-filled support. Every frequency inside
            #              the support is kept at full amplitude, including those where the
            #              noise is comparable to the signal.
            #   'wiener' : the frequencies are weighted by the estimated signal-to-noise
            #              ratio instead of being kept or dropped, and the support is the
            #              radius at which that weight falls below filter_threshold.
            if "filter_mode" not in v:
                v["filter_mode"] = "mask"
            if v["filter_mode"] not in ["mask", "wiener"]:
                raise ValueError(f"Invalid value for filter_mode. It is {v['filter_mode']} but should be mask or wiener")

            # Weight below which the filter is truncated to zero.
            if "filter_threshold" not in v:
                v["filter_threshold"] = 0.2

            # Frequency band, in units of the diffraction cutoff, used to calibrate the
            # noise level of each patch. It has to lie beyond the diffraction cutoff,
            # where the telescope cannot transmit any signal.
            if "filter_noise_band" not in v:
                v["filter_noise_band"] = [0.9, 1.45]

            # Shape of the noise power spectrum.
            #   'white' : flat, the usual assumption.
            #   'auto'  : measured from the patches themselves. Interpolating the frames
            #             (destretching) correlates the noise and makes it strongly
            #             non-white, so that a level calibrated beyond the diffraction
            #             cutoff badly underestimates the noise at lower frequencies.
            if "filter_noise_shape" not in v:
                v["filter_noise_shape"] = "white"
            if v["filter_noise_shape"] not in ["white", "auto"]:
                raise ValueError(f"Invalid value for filter_noise_shape. It is {v['filter_noise_shape']} but should be white or auto")

            # Frequency below which the measured noise shape is held flat. Below it the
            # measurement is contaminated by the signal of the patches themselves.
            if "filter_noise_flat" not in v:
                v["filter_noise_flat"] = 0.25

    if "psd" not in config:
        config["psd"] = {}
        config["psd"]["K"] = 100.0
        config["psd"]["v0"] = 0.1
        config["psd"]["p"] = 2.0
    # Retain priors block fallback if present for backward compatibility
    if "priors" not in config:
        config["priors"] = {}
        config["priors"]["K"] = {'mean': config["psd"]["K"], 'sigma': 1.0}
        config["priors"]["v0"] = {'mean': config["psd"]["v0"], 'sigma': 1.0}
        config["priors"]["p"] = {'mean': config["psd"]["p"], 'sigma': 1.0}
                                    
    return config