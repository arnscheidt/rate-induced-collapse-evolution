# simple many-agent evolutionary model, loosely based on Ferriere and Legendre (2013)
# originally run using Julia 1.5.3

############################
# INSTRUCTIONS 
############################
#
#
#
# RUNNING THE MODEL
#
# Initialize N agents with some initial x value using
# agents, births = initialize(N,x_initial)
#
# To step the model forward in time, first create two lists to save the evolution of types over time to, as follows:
#
# saved_types = Any[]
# saved_n = Any
#
# Then, run iterate_many!(agents, births...)
# See the function definition for the arguments. Brief summary here:
# evolution -> Boolean variable defining whether evolution is occurring or not
# T -> total number of timesteps
# Tsave -> the interval at which we want data saved to "saved_types" and "saved_n"
# params_start and params_end are lists of parameter values in the format [b,c,d,θ,h,w,a]
# If they are different, there will be linear ramping between the two.
#
# To summarize, an example run looks like
#
# agents, births = initialize(1000,100.0)
#
# saved_types = Any[]
# saved_n = Any[]
#
# iterate_many!(agents,births,true,saved_types,saved_n,100000,10000,[1.0e-5,5.0e-5,0.01,0.01,2.05,0.4,-9.16],[1.0e-5,5.0e-5,0.01,0.01,2.05,0.4,-9.16])
#
#
#
#
# PLOTTING, READING, AND WRITING
#
# saved_types and saved_n can be plotted using plot_types(), and saved to .csv using write_data(). Conversely, they can be extracted from previously saved .csv files using 
# saved_types, saved_births = read_data()
#
# A typical model run involves letting the system run to quasi-equilibrium, perturbing it using a linear parameter ramp, and then running the model for some time with the parameters at their new values. Each of these requires separate iterate_many! calls. 
#
#
#
#
# CRITICAL RATES FOR EXTINCTION
#
# Finally, the experiment on critical rates for inducing extinction is conducted using 
# critical_rate(ratemin,ratemax,nrate,Tmin,Tmax,nT,Ttotal,samples)
#
# The parameters are pretty self-explanatory:
# ratemin, ratemax -> minimum and maximum rates of change considered
# nrate -> sampling interval between ratemin and ratemax
# Tmin, Tmax -> minimum and maximum timescales considered
# nT -> sampling interval between Tmin and Tmax
# Ttotal -> the total amount of time to let the simulation run for. Should be longer than Tmax (extinction sometimes occurs with a small delay)
# samples is the number of times to run the model for each rate,T pair.
# 
# An example of this looks like 
# critical_rate(1e-9,1e-6,10,100,100000,10,200000,10)
#
# The figures in the paper are from previous model simulations saved to .csv files; see graphs.py for more details
#
#
#
#


using Random
using BenchmarkTools
using DataStructures
using DelimitedFiles
using Distributions
using Plots

############################
# MODEL DEFINITION
############################

const pmut = 0.001 # mutation rate

function α(z::Float64,h::Float64,w::Float64,a::Float64)
	return 2*h*(1-1/(1+exp(-w*(z+a))))
end

function initialize(N::Int64,x_initial::Float64)
	# initialize list, linear time
	agents = Dict{Float64,Int64}()
	agents[x_initial] = N
	
	births = Dict{Float64,Int64}()
	births[x_initial] = 0

	# initialize a 'births' Dict, (will save on allocations later)
	return agents, births
end

function sum(agents::Dict{Float64,Int64}) 
	# sum all the agents and trait values 
	types = collect(agents)
	running_sum_x = 0
	running_sum_n = 0
	for i in 1:length(types)
		running_sum_x += types[i][1]*types[i][2]
		running_sum_n += types[i][2]
	end
	return running_sum_n,running_sum_x
end

# note the next two functions can generate values outside [0,1]
# this doesn't matter though, p<0 will become p=0,
# and p>1 will become p=1

function pdeath(x::Float64,N::Int64,b::Float64,c::Float64)
	p = b*x*(x+1) + c*N
	if p<0
		p = 0
	elseif p>1
		p = 1
	end
	return p
end

function pbirth(agents::Dict{Float64,Int64},x_i::Float64,sum::Float64,d::Float64,θ::Float64,h::Float64,w::Float64,a::Float64)
	# calculate sum of alpha
	types = collect(agents)
	α_running_sum = 0
	for i in 1:length(types)
		α_running_sum+=α(x_i-types[i][1],h,w,a)*types[i][2]
	end
	p = d*sum/(1+α_running_sum+θ*sum)
	if p<0
		p = 0
	elseif p>1
		p = 1
	end
	return p

end

function iterate!(agents::Dict{Float64,Int64},births::Dict{Float64,Int64},evolution::Bool,params::Array{Float64,1})

	# we store all our data in a Dict type=>number
	# typical size below 20
	
	# to save on allocations, we maintain a second dict of all the species
	# in the agents dict, but with all numbers as zero
	
	N,sum_term = sum(agents)

	b,c,d,θ,h,w,a = params

	# BIRTHS

	types = collect(agents)
	temp_type = 0.0

	for itype in 1:length(types)
		n_temp = types[itype][2]

		p = pbirth(agents,types[itype][1],sum_term,d,θ,h,w,a)

		# the number of births is distributed Bin(n_temp,p)
		n_births = rand(Binomial(n_temp,p))
		
		# the number of mutations is distributed Bin(n_births,pmut)
		n_mut = rand(Binomial(n_births,pmut))
		
		# the key for the existing type should already exist
		births[types[itype][1]]+=n_births-n_mut

		if evolution
			for imut in 1:n_mut
				temp_type = types[itype][1] + rand([-1.0,1.0])
				if haskey(births,temp_type)
					births[temp_type] += 1
				else
					births[temp_type] = 1
				end
			end
		end

	end

	# DEATHS
	for itype in 1:length(types)
		n_temp = types[itype][2]

		p = pdeath(types[itype][1],N,b,c)

		# the number of deaths per type is distributed Bin(n_temp,p)
		n_deaths = rand(Binomial(n_temp,p))

		if n_deaths==agents[types[itype][1]]
			delete!(agents,types[itype][1])
		else
			agents[types[itype][1]] -= n_deaths
		end

	end

	# remove any types with zero births, then merge agents and births dicts
	for (key,value) in births
		if value==0
			delete!(births,key)
		end
	end

	mergewith!(+,agents,births)
	
	# re-sync empty births dict to new agents dict
	for (key,value) in births
		if haskey(agents,key)
			births[key] = 0
		else
			delete!(births,key)
		end
	end

	for(key,value) in agents
		if !haskey(births,key)
			births[key]=0
		end
	end
	
end

function iterate_many!(agents::Dict{Float64,Int64},births::Dict{Float64,Int64},evolution::Bool,saved_types,saved_n,T::Int,Tsave::Int,params_start::Array{Float64,1},params_end::Array{Float64,1})

	# T is the total number of timesteps
	# Tsave is the number of timesteps between saves

	#saved_types = Any[]
	#saved_n = Any[]
	# [1.0e-5,5.0e-5,0.01,0.01,2.05,0.4,-9.16]

	# a bit wasteful, but this is far from the limiting factor so it doesn't matter
	push!(saved_types,types(agents)[1])
	push!(saved_n,types(agents)[2])

	# calculate interval for linear changes
	params_int = (params_end-params_start)/T
	params_current = params_start[:]
	
	Nsave = floor(T/Tsave) # number of total saves

	for iN in 1:Nsave
		for iT in 1:Tsave
			iterate!(agents,births,evolution,params_current)
			params_current+=params_int
		end
		push!(saved_types,types(agents)[1])
		push!(saved_n,types(agents)[2])

	end
	return saved_types,saved_n
end

function types(agents::Dict{Float64,Int64})
	temp_array = collect(agents)
	
	Ntypes = size(temp_array)[1]
        types_array = zeros(Ntypes)
        n_array = zeros(Ntypes)

        for i in 1:Ntypes
                types_array[i] = temp_array[i][1]
                n_array[i] = temp_array[i][2]
        end

        return types_array,n_array

end

############################
# HELPER FUNCTIONS 
############################

function plot_types(saved_types,saved_n,Tstart,Tsave)
	
	scatter(Tstart*ones(size(saved_types[1])[1]),saved_types[1],zcolor=saved_n[1],m=(:Greens,5),legend=false)
	for i in 2:size(saved_types)[1]
		scatter!((Tstart+i*Tsave)*ones(size(saved_types[i])[1]),saved_types[i],zcolor=saved_n[i],m=(:Greens,5),legend=false)
	end
	gui()	
end

function write_data(saved_types,saved_n,filename::String)
	# write to dlm file
	writedlm(filename*"_types.csv",saved_types,',')
	writedlm(filename*"_n.csv",saved_n,',')
end

function read_data(filename::String)
	# read data
	rawtypes = readdlm(filename*"_types.csv",'\n')
	rawn = readdlm(filename*"_n.csv",'\n')

	# load histories
	
	types = Any[]
	n = Any[]
	for i in 1:size(rawtypes)[1]
		if typeof(rawtypes[i])==Float64
			push!(types,[rawtypes[i]])
		else 
			push!(types,parse.(Float64,split(rawtypes[i],",")))
		end

		if typeof(rawn[i])==Float64
			push!(n,[Int64(rawn[i])])
		else 
			push!(n,Int64.(parse.(Float64,split(rawn[i],","))))
		end

	end
	
	# generate agents and births from last row of data
	lastindex = size(types)[1]
	agents = Dict{Float64,Int64}()
	births = Dict{Float64,Int64}()

	for i_type in 1:size(types[lastindex])[1]
		agents[types[lastindex][i_type]] = n[lastindex][i_type]
		births[types[lastindex][i_type]] = 0
	end

	return types,n,agents,births
	# return types_array,n_array
end

############################
# EXPERIMENT SPECIFICATION
############################
function extinction_probability(T,Tmax,params_start,params_end,samples::Int64)
	# calculate the probability of extinction for the given perturbation
	initial_types, initial_n, initial_agents, initial_births = read_data("qess_hires")

	extinct = 0
	for i in 1:samples
		agents = copy(initial_agents)
		births = copy(initial_births)
		st = Any[]
		sn = Any[]

		# linear ramp over time T
		iterate_many!(agents,births,true,st,sn,T,T,params_start,params_end)

		# keep things the same for time Tmax - T to
		# check for delayed extinction
		iterate_many!(agents,births,true,st,sn,Tmax-T,Tmax-T,params_end,params_end)

		if isempty(agents)
			extinct += 1
		end
	end

	return extinct/samples
end

function critical_rate(ratemin,ratemax,nrate,Tmin,Tmax,nT,Ttotal,samples)
	# calculate probability of extinction for each rate-time pair
		
	# this assumes we are modifying the parameter b only
	
	# generate even log scale spacing
	rate_array = exp.(LinRange(log(ratemin),log(ratemax),nrate))
	T_array = Int.(floor.(exp.(LinRange(log(Tmin),log(Tmax),nT))))

	p = Array{Float64}(undef,nrate,nT)
	
	params_start = [1.0e-5,5.0e-5,0.01,0.01,2.05,0.4,-9.16]

	for irate in 1:nrate
		for iT in 1:nT
			params_end = [1.0e-5+rate_array[irate]*T_array[iT],5.0e-5,0.01,0.01,2.05,0.4,-9.16]
			p[irate,iT] = extinction_probability(T_array[iT],Ttotal,params_start,params_end,samples)
			print(irate,iT)
		end
	end
	return p	
end
