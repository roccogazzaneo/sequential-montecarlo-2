import particles
import particles.smc_samplers as ssp
import particles.state_space_models as ssm
import particles.distributions as dists


x, y = ssm.StochVol().simulate(100) # simulate data (default parameters)

# at this stage we have x and y simulated from the stochastic volatility model (x is a normal with expectation the equation)

# a data structure that allows you to organize info of different distribution in a organized structure
# for example calling rvs on this will sample both from mu and sigma distribution and save that in 2 arrays in a dict

prior = dists.StructDist({'mu': dists.Normal(scale=10.), 'sigma': dists.Gamma()})
prior_sampled = prior.rvs(size=5)
#print(prior_sampled['mu'])
#print(prior_sampled['sigma'])

fk_smc2 = ssp.SMC2(ssm_cls=ssm.StochVol, data=y, prior=prior, init_Nx=60, ar_to_increase_Nx=0.1)

alg_smc2 = particles.SMC(fk=fk_smc2, N=10)

alg_smc2.run()

print(alg_smc2)

