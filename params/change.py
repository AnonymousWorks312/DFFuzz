from params.parameters import Parameters

change = Parameters()

# ---------fidelity_mode---------
# change.fidelity_mode = 'none'
# change.fidelity_mode = 'ssim'
# change.fidelity_mode = 'euclidean'
# change.fidelity_mode = 'gan'
# change.fidelity_mode = 'dcgan'
change.fidelity_mode = 'acgan'

# ---------mutation_strategy_mode---------
change.mutation_strategy_mode = 'MCMC'
# change.mutation_strategy_mode = 'Roulette'
# change.mutation_strategy_mode = 'Random'
