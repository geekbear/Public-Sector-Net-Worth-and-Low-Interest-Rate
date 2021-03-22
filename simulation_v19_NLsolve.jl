#clearconsole()
#
# using NLsolve to solve transition dynamics

# parameterization
α = 1/3            # share of private capital in production function
r = 0.02           # global interest rate
τ = 0.1            # tax rate
δ = 0.06           # depreciation of public capital
δ_pr = 0.06        # depreciation of private capital
β = 0.98           # discount rate
ω = 50.0            # weight on public service in the utility function
b̄ = 0.4            # debt limit in share of GDP (net financial debt)
A = 1              # productivity of public capital - affecting its value q
ϕ = 0.5           # capital adjustment cost parameter

Φ = ((r+δ_pr)/α)^(α/(α-1))*A


# parameter restrictions
@assert τ*(1-α) - (r+δ)*b̄ > 0
@assert b̄*Φ < 1         # debt be less than 100 percent of output

# Equilibrium under a binding debt limit
using NLsolve
function debt_limit(T=100)

    # computing growth rate
    mu(g) = 1 + ϕ*(g+δ)
    LHS(g) = ω/β*(1+g)*(mu(g)-b̄*Φ) - ω*(ϕ/2*(g+δ)^2+(τ*(1-α)-b̄*(1+r))*Φ+mu(g)*(1-δ))
    RHS(g) = ((g-r)*b̄+τ*(1-α))*Φ - (g+δ)*(1+ϕ/2*(g+δ))
    res(g) = LHS(g) - RHS(g)
    #ini_guess = [β*(1+r)-1]        # note the vector representatio
    ini_guess = [0.0]
    sol=nlsolve(n_ary(res),ini_guess)
    @assert NLsolve.converged(sol) == true
    gᵈ = sol.zero[1]      # this turns sol.zero from a vector to a number
    #@assert gᵈ>r

    # generate vectors of key variables
    X_ini = 1.0           # initial capital stock, at time (-1)
    B_ini = b̄*Φ*X_ini     # initial debt level
    G_ini = RHS(gᵈ)*X_ini
    I_ini = (gᵈ+δ)*X_ini

    X_d = zeros(T+2)
    B_d = similar(X_d)
    G_d = similar(X_d)
    I_d = similar(X_d)

    for t in 1:T+2
        X_d[t] = X_ini*(1+gᵈ)^(t-1)       # first element is time -1 in model
        B_d[t] = B_ini*(1+gᵈ)^(t-1)
        G_d[t] = G_ini*(1+gᵈ)^(t-1)
        I_d[t] = I_ini*(1+gᵈ)^(t-1)
    end

    return I_d,G_d,X_d,B_d,gᵈ
end
I_d,G_d,X_d,B_d,gᵈ= debt_limit()    # default argument T
@show gᵈ

X_d_Y = 1/Φ*ones(length(X_d))
B_d_Y = b̄*ones(length(X_d))
G_d_Y = G_d ./ X_d /Φ
I_d_Y = I_d ./ X_d /Φ
growth_d = X_d[2:end]./X_d[1:end-1] .-1
## Equilibrium under networth growth rule
# BGP under net worth rule

temp = r+δ-τ*(1-α)*Φ
G_X_BGP(g,q) = ω*(temp + (q-1)*(1+r-(1+g)/β) + ϕ*(g+δ)*((1+g)/β-1-δ-(g+δ)/2))
B_X_BGP(g,q) = (τ*(1-α)*Φ-(g+δ)*(1+ϕ*(g+δ)/2)-G_X_BGP(g,q))/(r-g)

using Plots
plot(B_X_BGP,[1,1.5])
plot!(B_X_d,[1,1.5])


#---
# transition dynamics
using NLsolve

function networth_rule(T=100;g,q)

    # parameter restrictions
    @assert r<g

    # steady state and initial values
    η_ss = ω*(temp + (q-1)*(1+r-(1+g)/β) + ϕ*(g+δ)*((1+g)/β-1-δ-(g+δ)/2))
    b_ss = (τ*(1-α)*Φ-(g+δ)*(1+ϕ*(g+δ)/2)-η_ss)/(r-g)
    x_ss = 1+g
    i_ss = g+δ
    # check whether s.s. values satisfy the dynamic equations

    # initial value before the switch
    b0 = b̄*Φ
    η0 = G_d_Y[1]/X_d_Y[1]
    x0 = 1+gᵈ
    i0 = gᵈ+δ
    # starting value at the switch
    b1 = b0
    x1 = x0

    # Solving transition dynamics
    # set up the system of nonlinear equations
    function transition(vars)    # transition dynamics

        # vars is x2,...,xT;b2,...bT;η1,...ηT;i1,..,iT length 4T-2
        # x1 and b1 are given and not be solved.

        x = vcat(x1,vars[1:T-1])                                                                                                                                            )
        b = vcat(b1,vars[T:2T-2])      # now length T, B/X
        η = vars[2*T-1:3*T-2]              # length T, G/X
        i = vars[3*T-1:end]              # end is 4T-1. length of ix is T as well.

        # equations governing transition, 4T-2 equations in total
        FOC = zeros(T)               # FOC, column vector
        bud_cons = zeros(T)          # budget constraint
        nw_cons = zeros(T-1)         # networth rule
        cap = zeros(T-1)             # law of motion for capital

        for t = 1:T-1
            FOC[t] = η[t+1] + η[t+1]*x[t+1]/η[t]*ω/β*(q-1-ϕ*i[t]) + ω*(ϕ/2*i[t+1]^2+τ*(1-α)*Φ+(1+ϕ*i[t+1]*(1-δ))-q*(1+r))
            bud_cons[t] = b[t+1]*x[t+1] - η[t] - i[t]*(1+ϕ*i[t]/2) - τ*(1-α)*Φ + (1+r)*b[t]
            nw_cons[t] = (q-b[t+1])*z[t] - (1+g)*(q-b[t])
            cap[t] = x[t+1] - (1-δ) - i[t]
        end     # 4*(T-1) equations
        #assuming s.s. is reached at T+1. such that b[T+1]=b_ss, η[T+1]=η_ss,...
        FOC[T] = η_ss + η_ss*x_ss/η[T]*ω/β*(q-1-ϕ*i[T]) + ω*(ϕ/2*i_ss^2+τ*(1-α)*Φ+(1+ϕ*i_ss*(1-δ))-q*(1+r))
        bud_cons[T] = b_ss*x_ss - η[T] - i[t]*(1+ϕ*i[T]/2) - τ*(1-α)*Φ + (1+r)*b[T]

        res = vcat(FOC[1:T], bud_cons[1:T], nw_cons[1:T-1], cap[1:T-1])

        return res
    end

    function jacobian(x)

        x = vcat(x1,vars[1:T-1])                                                                                                                                            )
        b = vcat(b1,vars[T:2T-2])      # now length T, B/X
        η = vars[2*T-1:3*T-2]              # length T, G/X
        i = vars[3*T-1:end]              # end is 4T-1. length of ix is T as well.

        FOC_b = zeros(T,T)             # derivative of FOC w.r.t. b
        FOC_η = zeros(T,T)
        FOC_x = zeros(T,T)
        FOC_i = zeros(T,T)

        bud_cons_b = zeros(T,T)
        bud_cons_η = zeros(T,T)
        bud_cons_x = zeros(T,T)
        bud_cons_i = zeros(T,T)

        nw_cons_b = zeros(T-1,T)
        nw_cons_η = zeros(T-1,T)
        nw_cons_x = zeros(T-1,T)
        nw_cons_i = zeros(T-1,T)

        cap_b = zeros(T-1,T)
        cap_η = zeros(T-1,T)
        cap_x = zeros(T-1,T)
        cap_i = zeros(T-1,T)

        for t in 1:T-1

            FOC_η[t,t] = -ω/β*(q-1-ϕ*i[t])*η[t+1]*x[t+1]/(η[t]^2)
            FOC_η[t,t+1] = 1 + x[t+1]/η[t]*ω/β*(q-1-ϕ*i[t])
            FOC_x[t,t+1] = η[t+1]/η[t]*ω/β*(q-1-ϕ*i[t])
            FOC_i[t,t] = -η[t+1]/η[t]*x[t+1]*ω/β*ϕ
            FOC_i[t,t+1] = ω*(ϕ*i[t+1]+ϕ*(1-δ))

            bud_cons_b[t,t] = -(1+r)
            bud_cons_b[t,t+1] = x[t+1]
            bud_cons_η[t,t] = -1
            bud_cons_x[t,t+1] = b[t+1]
            bud_cons_i[t,t] = -1-ϕ*i[t]

            nw_cons_b[t,t] = 1+g
            nw_cons_b[t,t+1] = -x[t+1]
            nw_cons_x[t,t+1] = -b[t+1]

            cap_x[t,t+1] = 1
            cap_i[t,t] = -1
        end

        FOC_η[T,T] = -ω/β*(q-1-ϕ*i[T])*η_ss*x_ss/(η[T]^2)
        FOC_x[T,T] = 0
        FOC_i[T,T] =  -η_ss/η[T]*x_ss*ω/β*ϕ

        bud_cons_b[T,T] = -(1+r)
        bud_cons_η[T,T] = -1
        bud_cons_i[T,T] = -1-ϕ*i[T]

        cap_i[T,T] = -1
        # note that there are only T-1 nw_cons equations

        # put together the building blocks
        FOC = hcat(FOC_b[:,2:end],FOC_η,FOC_x,FOC_i)
        bud_cons = hcat(bud_cons_b[:,2:end],bud_cons_η,bud_cons_x,bud_cons_i)
        nw_cons = hcat(nw_cons_b[:,2:end],nw_cons_η,nw_cons_x,nw_cons_i)
        cap = hcat(cap_b[:,2:end],cap_η,cap_x,cap_i)

        jacobian = vcat(FOC,bud_cons,nw_cons,cap)    # finally the Jacobian matrix

        return jacobian

    end

    # initial guess for b,j
    b_guess = Vector(range(b_ini,b_ss,length=T-1))     # can also use "collect"
    η_guess = Vector(range(η_ini,η_ss,length=T))
    x_guess = Vector(range(x_ini,x_ss,length=T))
    i_guess = Vector(range(i_ini,i_ss,length=T))
    guess = vcat(x_guess,b_guess,η_guess,i_guess)

    # solving the system of equations
    sol = nlsolve(transition, jacobian, guess)
    #@assert NLsolve.converged(sol) == true
    @show NLsolve.converged(sol)
    @show sol.iterations
    b_retrieve = sol.zero[1:T-1]
    b = vcat(b_ini,b_retrieve)
    η = sol.zero[T:2*T-1]
    x = sol.zero[2*T:3*T-1]
    i = sol.zero[3*T:end]

    # retrieve the G,X,B,I sequences
    X = zeros(T+1)         # length T+1 since it includes the starting point, and T solved periods
    B = similar(X)
    G = similar(X)
    I = similar(X)

    X[1:2] = X_d[1:2]          # the first element is X_-1 in the model
    B[1:2] = B_d[1:2]
    G[1] = G_d[1]
    I[1] = X[2] - (1-δ)*X[1]
    for t in 2:T
        X[t+1] = z[t-1]*X[t]     # X[3] = X[2]*z[1];
        G[t] = j[t-1]*X[t]       # G[2] = X[2]*j[1]
        B[t] = b[t-1]*X[t]         # B[2] = X[2]*b[1]
        I[t] = X[t+1] - (1-δ)*X[t]
    end

    return I,G,X,B,z,j,b,q,g

end

I_nw,G_nw,X_nw,B_nw,z,j,b,q,g = networth_rule(;g=gᵈ,q=1.06)

X_nw_Y = 1/Φ*ones(length(X_nw))
G_nw_Y = G_nw ./ X_nw /Φ
I_nw_Y = I_nw ./ X_nw /Φ
B_nw_Y = B_nw ./ X_nw /Φ

# check whether the series satisfy conditions
#networth_check = q*X_nw[100] - B_nw[100] - (1+g)*(q*X_nw[99]-B_nw[99])
#combined_check = (q-1)*X_nw[2]-G_nw[1]-((1+r)*(q-1)+temp)*X_nw[1]+(r-g)*(q*X_nw[1]-B_nw[1])
#FOC_check = (q-1)*ω/G_nw[99] + β*(1/X_nw[100]-ω/G_nw[100]*((q-1)*(1+r)+temp))

#---
using Plots
pyplot()
function makeplots(I_d,G_d,X_d,B_d,I_nw,G_nw,X_nw,B_nw,T=100)

    #pyplot()          # backend(); default backend is GR

    p1 = plot(-1:T-2,[I_d[1:T],I_nw[1:T]],label=["debt limit" "combined rules"],grid=false);
    xlabel!("Time",guidefontsize=8)
    title!("Public investment",titlefontsize=10)
    #ylims!(0,0.015)
    p2 = plot(-1:T-2,[G_d[1:T],G_nw[1:T]],legend=false,grid=false);
    xlabel!("Time",guidefontsize=8)
    title!("Non-interest current expenditure",titlefontsize=10)
    #ylims!(0.05,0.20)
    p3 = plot(-1:T-2,[X_d[1:T],X_nw[1:T]],legend=false,grid=false);
    xlabel!("Time",guidefontsize=8)
    title!("Public capital stock",titlefontsize=10)
    #ylims!(0.5,1.3)
    p4 = plot(-1:T-2,[B_d[1:T],B_nw[1:T]],legend=false,grid=false);
    xlabel!("Time",guidefontsize=8)
    title!("Public debt",titlefontsize=10)
    #ylims!(0,0.5)
    plot(p1,p2,p3,p4,layout=(2,2))

end

function makeplots_ratio(I_d,G_d,growth_d,B_d,I_nw,G_nw,growth,B_nw,T=100)

    #pyplot()          # backend(); default backend is GR

    p1 = plot(-1:T-2,[I_d[1:T],I_nw[1:T]],label=["debt limit" "combined rule"],grid=false);
    xlabel!("Time",guidefontsize=8)
    title!("Public investment",titlefontsize=10)
    #ylims!(0,0.015)
    p2 = plot(-1:T-2,[G_d[1:T],G_nw[1:T]],legend=false,grid=false);
    xlabel!("Time",guidefontsize=8)
    title!("Non-interest current expenditure",titlefontsize=10)
    #ylims!(0.05,0.20)
    p3 = plot(-1:T-2,[growth_d[1:T],growth[1:T]],legend=false,grid=false);
    xlabel!("Time",guidefontsize=8)
    title!("Output growth rate",titlefontsize=10)
    #ylims!(0.5,1.3)
    p4 = plot(-1:T-2,[B_d[1:T],B_nw[1:T]],legend=false,grid=false);
    xlabel!("Time",guidefontsize=8)
    title!("Public debt",titlefontsize=10)
    #ylims!(0,0.5)
    plot(p1,p2,p3,p4,layout=(2,2))

end

makeplots(I_d,G_d,X_d,B_d,I_nw,G_nw,X_nw,B_nw,30)
savefig("switch with same g and q=$q .png")

makeplots_ratio(I_d_Y,G_d_Y,growth_d,B_d_Y,I_com_Y,G_com_Y,growth,B_com_Y,40)   # in percent of GDP
savefig("combined rule in ratios.png")

function makeplots_com(I_d,G_d,X_d,B_d,I,G,X,B,T=100)

    #pyplot()          # backend(); default backend is GR

    p1 = plot(-1:T-2,[I_d[1:T],I[1:T]],label=["debt limit" "combined rule"],grid=false);
    xlabel!("Time",guidefontsize=8)
    title!("Public investment",titlefontsize=10)
    #ylims!(0,0.015)
    p2 = plot(-1:T-2,[G_d[1:T],G[1:T]],legend=false,grid=false);
    xlabel!("Time",guidefontsize=8)
    title!("Non-interest current expenditure",titlefontsize=10)
    #ylims!(0.05,0.20)
    p3 = plot(-1:T-2,[X_d[1:T],X[1:T]],legend=false,grid=false);
    xlabel!("Time",guidefontsize=8)
    title!("Public capital stock",titlefontsize=10)
    #ylims!(0.5,1.3)
    p4 = plot(-1:T-2,[B_d[1:T],B[1:T]],legend=false,grid=false);
    xlabel!("Time",guidefontsize=8)
    title!("Public debt",titlefontsize=10)
    #ylims!(0,0.5)
    plot(p1,p2,p3,p4,layout=(2,2))

end

makeplots_com(I_d,G_d,X_d,B_d,I_com,G_com,X_com,B_com,10)
savefig("Combined rule")
makeplots_com(I_d_Y,G_d_Y,X_d_Y,B_d_Y,I_com_Y,G_com_Y,X_com_Y,B_com_Y,40)
