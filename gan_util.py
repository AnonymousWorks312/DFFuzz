
def set_gan_mode(fidelity_mode, dataset):
    print("***fidelity_mode***", fidelity_mode)
    if fidelity_mode == "gan":
        from gan.GAN_utils import GAN
        return GAN(dataset)
    elif fidelity_mode == "acgan":
        from acgan.ACGAN_utils import ACGAN
        return ACGAN(dataset)
    else:
        from dcgan.DCGAN_utils import DCGAN
        return DCGAN(dataset)