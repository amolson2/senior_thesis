earth_radius = 6378E3
mu = 3.986E14

# initial orbit
h1 = 2000E3
r1 = earth_radius + h1
v_i = (mu/r1)**0.5
# print(v_i)

# final orbit
h2 = 3000E3
r2 = earth_radius + h2
v_f = (mu/r2)**0.5
# print(v_f)

# delta-v
v_p = (2*mu*((1/r1)-(1/(r1+r2))))**0.5
delta_v1 = v_p - v_i
print(delta_v1)
delta_v2 = v_f*(1-((2*r1)/(r1+r2))**0.5)
print(delta_v2)

delta_v_tot = delta_v1 + delta_v2
print(delta_v_tot)

