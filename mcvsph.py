import taichi as ti
from sph_base import SPHBase
import numpy as np
import time


@ti.data_oriented
class MCVSPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        self.turb_mode = 3  ## 0 - DFSPH  1 - VRSPH  2 - micropolar 3 - Monte Carlo vortex particles
        self.show_curl_color = True
        self.show_sample_color = False
        self.exponent = 7.0
        self.exponent = self.ps.cfg.get_cfg("exponent")

        self.stiffness = 50000.0
        self.stiffness = self.ps.cfg.get_cfg("stiffness")

        self.surface_tension = 0.01
        self.dt[None] = self.ps.cfg.get_cfg("timeStepSize")

        self.is_propeller = False
        self.is_propeller = self.ps.cfg.get_cfg("is_propeller")

        if self.is_propeller:
            self.propeller_rot_mat = ti.Matrix.field(3,
                                                     3,
                                                     dtype=float,
                                                     shape=())
            self.propeller_speed = 8.0 * ti.math.pi
            self.generate_propeller_rot_mat(self.propeller_speed)
            rigid_bodies = self.ps.cfg.get_rigid_bodies()
            self.offset_propeller_np = np.array(rigid_bodies[0]['translation'])
            self.offset_propeller = ti.Vector([0.0 for _ in range(3)])
            for i in ti.static(range(3)):
                self.offset_propeller[i] = self.offset_propeller_np[i]

        self.no_turbulence = False
        self.enable_mpsph = False
        self.enable_vrsph = False
        self.enable_mcvsph = False
        self.enable_macro_micro_motion = False
        self.t_one_step = 0.0

        ##### DFSPH Setting ################
        self.m_max_iterations_v = 100
        self.m_max_iterations = 100

        self.m_eps = 1e-5
        self.max_error_V = 0.1
        self.max_error = 0.05
        self.inv_dt = 1.0 / self.dt[None]
        self.inv_dt2 = 1.0 / (self.dt[None] * self.dt[None])

        #### Micropolar Setting ############
        self.micro_inertia = 2.0
        self.transfer_coefficient = 0.15
        self.rotation_viscosity = 0.001
        self.inv_micro_inertia = 1.0 / self.micro_inertia
        self.vorticity = ti.Vector.field(self.ps.dim,
                                         dtype=float,
                                         shape=self.ps.particle_max_num)

        #### VRSPH setting ###################
        self.alpha = 1.2

        ###### Evaluation ##############
        self.total_vorticity = ti.field(dtype=ti.f64, shape=())
        self.kinetic_energy = ti.field(dtype=ti.f64, shape=())
        self.helicity = ti.field(dtype=ti.f64, shape=())
        self.helicity_abs = ti.field(dtype=ti.f64, shape=())
        self.rot_energy = ti.field(dtype=ti.f64, shape=())
        self.potential_energy = ti.field(dtype=ti.f64, shape=())
        self.total_energy = ti.field(dtype=ti.f64, shape=())
        self.total_energy_plus_rot_energy = ti.field(dtype=ti.f64, shape=())
        self.vorticity_norm_sum = ti.field(dtype=ti.f64, shape=())
        self.total_vorticity_buffer = []
        self.kinetic_energy_buffer = []
        self.helicity_buffer = []
        self.helicity_abs_buffer = []
        self.rot_energy_buffer = []
        self.potential_energy_buffer = []
        self.total_energy_buffer = []
        self.total_energy_plus_rot_energy_buffer = []
        self.vorticity_norm_sum_buffer = []
        self.it_num_div_buffer = []
        self.it_num_den_buffer = []
        self.streamFunc = ti.Vector.field(self.ps.dim,
                                          dtype=float,
                                          shape=self.ps.particle_max_num)
        self.inv_fluid_particle_num = 1.0 / self.ps.fluid_particle_num

        ########### Monte Carlo Setting ####################
        self.delta_vorticity = ti.Vector.field(self.ps.dim,
                                               dtype=float,
                                               shape=self.ps.particle_max_num)
        self.v_star = ti.Vector.field(self.ps.dim,
                                      dtype=float,
                                      shape=self.ps.particle_max_num)
        self.sample_num = int(self.ps.particle_max_num / 1000)
        #self.sample_num = int(self.ps.particle_max_num / 160)  #
        self.sample_num = self.generate_monte_carlo_samples(
        )  #remove solid samples
        print("sample_num =", self.sample_num)
        self.inv_sample_num = 1.0 / self.sample_num
        self.vortex_sph_particle_ratio = 0.05
        self.inv_prob_uni = self.ps.fluid_particle_num
        self.inv_prob_uni *= self.vortex_sph_particle_ratio * self.ps.m_V0

        self.monte_carlo_sample_index = ti.field(dtype=int,
                                                 shape=self.sample_num)
        self.delta_vorticity_monte_carlo_samples = ti.Vector.field(self.ps.dim,
                                                                   dtype=float)
        self.neighbor_particle_num = ti.field(dtype=int)
        self.monte_carlo_snode = ti.root.dense(ti.i, self.sample_num).place(
            self.delta_vorticity_monte_carlo_samples,
            self.neighbor_particle_num)

        print("fluid_particle_num =", self.ps.fluid_particle_num)

        if self.turb_mode == 0:
            self.no_turbulence = True
            self.enable_mpsph = False
            self.enable_vrsph = False
            self.enable_mcvsph = False
            self.enable_macro_micro_motion = False
        elif self.turb_mode == 1:
            self.no_turbulence = False
            self.enable_mpsph = False
            self.enable_vrsph = True
            self.enable_mcvsph = False
            self.enable_macro_micro_motion = False
        elif self.turb_mode == 2:
            self.no_turbulence = False
            self.enable_mpsph = True
            self.enable_vrsph = False
            self.enable_mcvsph = False
            self.enable_macro_micro_motion = False
        elif self.turb_mode == 3:
            self.no_turbulence = False
            self.enable_mpsph = False
            self.enable_vrsph = False
            self.enable_mcvsph = True
            self.enable_macro_micro_motion = False

    @ti.kernel
    def generate_propeller_rot_mat(self, speed: float):
        rot_mat4 = ti.math.rotation3d(0.0, 0.0, speed * self.dt[None])
        for i, j in ti.static(ti.ndrange(3, 3)):
            self.propeller_rot_mat[None][i, j] = rot_mat4[i, j]

    @ti.kernel
    def compute_DFSPH_factor(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            sum_grad_p_k = 0.0
            grad_p_i = ti.Vector([0.0 for _ in range(self.ps.dim)])
            # `ret` concatenates `grad_p_i` and `sum_grad_p_k`
            ret = ti.Vector([0.0 for _ in range(self.ps.dim + 1)])
            self.ps.for_all_neighbors(p_i, self.compute_DFSPH_factor_task, ret)
            sum_grad_p_k = ret[3]
            for i in ti.static(range(3)):
                grad_p_i[i] = ret[i]
            sum_grad_p_k += grad_p_i.norm_sqr()
            # Compute pressure stiffness denominator
            factor = 0.0
            if sum_grad_p_k > 1e-6:
                factor = 1.0 / sum_grad_p_k
            else:
                factor = 0.0
            self.ps.dfsph_factor[p_i] = factor

    @ti.func
    def compute_DFSPH_factor_task(self, p_i, p_j, ret: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            grad_p_j = self.ps.m[p_j] * self.cubic_kernel_derivative(
                self.ps.x[p_i] - self.ps.x[p_j])
            ret[3] += grad_p_j.norm_sqr()  # sum_grad_p_k
            for i in ti.static(range(3)):  # grad_p_i
                ret[i] += grad_p_j[i]
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            grad_p_j = self.ps.m_V[
                p_j] * self.density_0 * self.cubic_kernel_derivative(
                    self.ps.x[p_i] - self.ps.x[p_j])
            for i in ti.static(range(3)):  # grad_p_i
                ret[i] += grad_p_j[i]

    @ti.func
    def compute_densities_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            x_j = self.ps.x[p_j]
            ret += self.ps.m[p_j] * self.cubic_kernel((x_i - x_j).norm())
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            x_j = self.ps.x[p_j]
            ret += self.ps.m_V[p_j] * self.density_0 * self.cubic_kernel(
                (x_i - x_j).norm())

    @ti.kernel
    def compute_densities(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.ps.density[p_i] = self.ps.m[p_i] * self.cubic_kernel(0.0)
            den = 0.0
            self.ps.for_all_neighbors(p_i, self.compute_densities_task, den)
            self.ps.density[p_i] += den
            #self.ps.density[p_i] = ti.max(self.ps.density[p_i],self.density_0)
            self.ps.m_V[p_i] = self.ps.m[p_i] / self.ps.density[p_i]

    @ti.func
    def compute_non_pressure_forces_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]

        ############## Surface Tension ###############
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            diameter2 = self.ps.particle_diameter * self.ps.particle_diameter
            x_j = self.ps.x[p_j]
            r = x_i - x_j
            r2 = r.dot(r)
            if r2 > diameter2:
                ret -= self.surface_tension / self.ps.m[p_i] * self.ps.m[
                    p_j] * r * self.cubic_kernel(r.norm())
            else:
                ret -= self.surface_tension / self.ps.m[p_i] * self.ps.m[
                    p_j] * r * self.cubic_kernel(
                        ti.Vector([self.ps.particle_diameter, 0.0, 0.0
                                   ]).norm())

        ############### Viscosity Force ###############
        d = 2 * (self.ps.dim + 2)
        x_j = self.ps.x[p_j]
        # Compute the viscosity force contribution
        r = x_i - x_j
        v_xy = (self.ps.v[p_i] - self.ps.v[p_j]).dot(r)

        if self.ps.material[p_j] == self.ps.material_fluid:
            f_v = d * self.viscosity * self.ps.m_V[p_j] * v_xy / (
                r.norm()**2 + 0.01 *
                self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
            ret += f_v
        elif self.ps.material[p_j] == self.ps.material_solid:
            boundary_viscosity = 0.0
            # Boundary neighbors
            ## Akinci2012
            f_v = d * boundary_viscosity * (
                self.density_0 * self.ps.m_V[p_j] /
                (self.ps.density[p_i])) * v_xy / (
                    r.norm()**2 + 0.01 * self.ps.support_radius**2
                ) * self.cubic_kernel_derivative(r)
            ret += f_v
            if self.ps.is_dynamic_rigid_body(p_j):
                self.ps.acceleration[
                    p_j] += -f_v * self.ps.density[p_i] / self.ps.density[p_j]

    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0.0)
                continue
            ############## Body force ###############
            # Add body force
            d_v = ti.Vector(self.g)
            self.ps.acceleration[p_i] = d_v
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.for_all_neighbors(
                    p_i, self.compute_non_pressure_forces_task, d_v)
                self.ps.acceleration[p_i] = d_v

    @ti.kernel
    def advect(self):
        # Update position
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                if self.ps.is_dynamic_rigid_body(p_i):
                    self.ps.v[p_i] += self.dt[None] * self.ps.acceleration[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]

    @ti.kernel
    def predict_velocity(self):
        # compute new velocities only considering non-pressure forces
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i] and self.ps.material[
                    p_i] == self.ps.material_fluid:
                self.ps.v[p_i] += self.dt[None] * self.ps.acceleration[p_i]

    @ti.kernel
    def micropolar_solve(self):
        #update acceleration
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            ret = ti.Struct(curl_spin=ti.Vector(
                [0.0 for _ in range(self.ps.dim)]),
                            num_particles=0)
            self.ps.for_all_neighbors(p_i, self.compute_curl_spin_task, ret)
            #if ret.num_particles > 20:
            # self.ps.acceleration[p_i] += (self.transfer_coefficient +
            #                             self.viscosity) * ret.curl_spin
            self.ps.v[p_i] += self.dt[None] * (self.transfer_coefficient +
                                               self.viscosity) * ret.curl_spin
        #update spin
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] == self.ps.material_fluid:
                curl_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
                self.ps.for_all_neighbors(p_i,
                                          self.compute_vorticity_task_diff,
                                          curl_v)
                self.vorticity[p_i] = curl_v
                lap_spin = ti.Vector([0.0 for _ in range(self.ps.dim)])
                self.ps.for_all_neighbors(p_i, self.compute_lap_spin_task,
                                          lap_spin)
                lap_spin *= 2 * (self.ps.dim + 2)
                d_spin = self.inv_micro_inertia * (
                    (self.transfer_coefficient + self.viscosity) *
                    (curl_v - 2 * self.ps.spin[p_i]) +
                    self.rotation_viscosity * lap_spin)
                self.ps.spin[p_i] += self.dt[None] * d_spin
            elif self.ps.material[p_i] == self.ps.material_solid:
                self.ps.spin[p_i].fill(0)
                self.vorticity[p_i].fill(0)

    @ti.func
    def compute_vorticity_task_diff(self, p_i, p_j, curl_v: ti.template()):
        x_i = self.ps.x[p_i]
        x_j = self.ps.x[p_j]
        if self.ps.material[p_j] == self.ps.material_fluid:
            curl_v += self.ps.m_V[p_j] * (
                self.ps.v[p_j] - self.ps.v[p_i]).cross(
                    self.cubic_kernel_derivative(x_i - x_j))
        elif self.ps.material[p_j] == self.ps.material_solid:
            curl_v += self.ps.m_V[p_j] * (
                self.ps.v[p_j] - self.ps.v[p_i]).cross(
                    self.cubic_kernel_derivative(x_i - x_j))

    @ti.func
    def compute_lap_vorticity_task(self, p_i, p_j, lap_omega: ti.template()):
        x_i = self.ps.x[p_i]
        x_j = self.ps.x[p_j]
        r = x_i - x_j
        if self.ps.material[p_j] == self.ps.material_fluid:
            lap_omega += self.ps.m_V[p_j] * (
                self.vorticity[p_i] -
                self.vorticity[p_j]).dot(r) * self.cubic_kernel_derivative(
                    r) / (r.norm()**2 + 0.01 * self.ps.support_radius**2)
        elif self.ps.material[p_j] == self.ps.material_solid:
            lap_omega += self.ps.m_V[p_j] * (
                self.vorticity[p_i] -
                self.vorticity[p_j]).dot(r) * self.cubic_kernel_derivative(
                    r) / (r.norm()**2 + 0.01 * self.ps.support_radius**2)

    @ti.func
    def compute_lap_spin_task(self, p_i, p_j, lap_spin: ti.template()):
        x_i = self.ps.x[p_i]
        x_j = self.ps.x[p_j]
        r = x_i - x_j
        if self.ps.material[p_j] == self.ps.material_fluid:
            lap_spin += self.ps.m_V[p_j] * (
                self.ps.spin[p_i] -
                self.ps.spin[p_j]).dot(r) * self.cubic_kernel_derivative(r) / (
                    r.norm()**2 + 0.01 * self.ps.support_radius**2)
        elif self.ps.material[p_j] == self.ps.material_solid:
            lap_spin += self.ps.m_V[p_j] * (
                self.ps.spin[p_i] -
                self.ps.spin[p_j]).dot(r) * self.cubic_kernel_derivative(r) / (
                    r.norm()**2 + 0.01 * self.ps.support_radius**2)

    @ti.func
    def compute_curl_spin_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        x_j = self.ps.x[p_j]
        if self.ps.material[p_j] == self.ps.material_fluid:
            ret.curl_spin += self.ps.m_V[p_j] * (
                self.ps.spin[p_j] - self.ps.spin[p_i]).cross(
                    self.cubic_kernel_derivative(x_i - x_j))
            ret.num_particles += 1
        elif self.ps.material[p_j] == self.ps.material_solid:
            ret.curl_spin += self.ps.m_V[p_j] * (
                self.ps.spin[p_j] - self.ps.spin[p_i]).cross(
                    self.cubic_kernel_derivative(x_i - x_j))
            ret.num_particles += 1

    @ti.kernel
    def compute_particles_color_curl(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] == self.ps.material_fluid:
                #if self.ps.is_in_dynamic_area[p_i] == True:
                #color_base = ti.Vector([0.196,0.392,0.784])
                color_base = ti.Vector([0.0, 0.0, 1.0])
                color_vis_curl = ti.Vector([0.0, 0.0, 0.0])
                v_curl = ti.Vector([0.0, 0.0, 0.0])
                self.ps.for_all_neighbors(
                    p_i, self.compute_particles_color_curl_task, v_curl)
                self.ps.vorticity_eva[p_i] = v_curl
                self.curl_color(v_curl, color_vis_curl)
                self.ps.particle_color[p_i] = ti.math.clamp(
                    color_base + color_vis_curl, 0.1, 1.0)
                # self.particle_color[p_i] = ti.Vector([
                #     0.1, 0.1, 1.0])
            # elif not self.ps.is_in_dynamic_area[p_i]:
            #     self.particle_color[p_i] = ti.Vector([0.0,1.0,0.0])
            elif self.ps.material[p_i] == self.ps.material_solid:
                self.ps.particle_color[p_i] = ti.Vector([0.9, 0.9, 0.0])
                self.ps.vorticity_eva[p_i].fill(0)
        if self.enable_mcvsph and self.show_sample_color:
            for i in range(self.sample_num):
                p_i = self.monte_carlo_sample_index[i]
                self.ps.particle_color[p_i] = ti.Vector([1.0, 0.0, 0.0])

    @ti.func
    def compute_particles_color_curl_task(self, p_i, p_j,
                                          curl_v: ti.template()):
        x_i = self.ps.x[p_i]
        x_j = self.ps.x[p_j]
        if self.ps.material[p_j] == self.ps.material_fluid:
            curl_v += self.ps.m_V[p_j] * (
                self.ps.v[p_j] - self.ps.v[p_i]).cross(
                    self.cubic_kernel_derivative(x_i - x_j))
        elif self.ps.material[p_j] == self.ps.material_solid:
            curl_v += self.ps.m_V[p_j] * (
                self.ps.v[p_j] - self.ps.v[p_i]).cross(
                    self.cubic_kernel_derivative(x_i - x_j))

    @ti.func
    def curl_color(self, v: ti.template(), w: ti.template()):
        v_norm = v.norm()
        w[0] = -ti.exp(-0.03 * v_norm) + 1
        w[1] = w[0]

    ########### compute omega per time step version  ########
    @ti.kernel
    def compute_delta_vorticity_for_sph_particles(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] == self.ps.material_fluid:
                # update v_adv
                self.v_star[p_i] = self.ps.v[
                    p_i] + self.ps.acceleration[p_i] * self.dt[None]
                #compute omega
                vorticity = ti.Vector([0.0 for _ in range(self.ps.dim)])
                self.ps.for_all_neighbors(p_i,
                                          self.compute_vorticity_task_diff,
                                          vorticity)
                self.vorticity[p_i] = vorticity
            elif self.ps.material[p_i] == self.ps.material_solid:
                self.vorticity[p_i].fill(0)
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] == self.ps.material_fluid:
                #compute curl_v_adv
                curl_v_star = ti.Vector([0.0 for _ in range(self.ps.dim)])
                self.ps.for_all_neighbors(
                    p_i, self.compute_vorticity_star_task_diff, curl_v_star)
                #compute vortex stretching term
                grad_v = ti.Matrix([[0.0 for _ in range(self.ps.dim)]
                                    for _ in range(self.ps.dim)])
                self.ps.for_all_neighbors(p_i, self.compute_grad_vel_task_diff,
                                          grad_v)
                vortex_stretching = self.vorticity[p_i] @ grad_v
                #compute rotation viscosity term
                lap_omega = ti.Vector([0.0 for _ in range(self.ps.dim)])
                self.ps.for_all_neighbors(p_i, self.compute_lap_vorticity_task,
                                          lap_omega)
                lap_omega *= 2 * (self.ps.dim + 2)
                rot_vis = self.viscosity * lap_omega
                #advect omega
                vorticity_star = self.vorticity[p_i] + (
                    vortex_stretching + rot_vis) * self.dt[None]
                #update delta_omega
                self.delta_vorticity[p_i] = vorticity_star - curl_v_star
            elif self.ps.material[p_i] == self.ps.material_solid:
                self.delta_vorticity[p_i].fill(0)

    @ti.func
    def compute_vorticity_star_task_diff(self, p_i, p_j,
                                         vorticity_star: ti.template()):
        x_i = self.ps.x[p_i]
        x_j = self.ps.x[p_j]
        if self.ps.material[p_j] == self.ps.material_fluid:
            vorticity_star += self.ps.m_V[p_j] * (
                self.v_star[p_j] - self.v_star[p_i]).cross(
                    self.cubic_kernel_derivative(x_i - x_j))
        elif self.ps.material[p_j] == self.ps.material_solid:
            vorticity_star += self.ps.m_V[p_j] * (
                self.v_star[p_j] - self.v_star[p_i]).cross(
                    self.cubic_kernel_derivative(x_i - x_j))

    @ti.func
    def compute_grad_vel_task_diff(self, p_i, p_j, grad_vel: ti.template()):
        x_i = self.ps.x[p_i]
        x_j = self.ps.x[p_j]
        if self.ps.material[p_j] == self.ps.material_fluid:
            term_1_vec = (self.v_star[p_j] - self.v_star[p_i])
            #term_1_vec = (self.ps.v[p_j]-self.ps.v[p_i])
            term_2_vec = self.cubic_kernel_derivative(x_i - x_j)
            term_1_mat = ti.Matrix([[term_1_vec[0]], [term_1_vec[1]],
                                    [term_1_vec[2]]])
            term_2_mat = ti.Matrix(
                [[term_2_vec[0], term_2_vec[1], term_2_vec[2]]])
            grad_vel += self.ps.m_V[p_j] * term_1_mat @ term_2_mat
        elif self.ps.material[p_j] == self.ps.material_solid:
            term_1_vec = (self.v_star[p_j] - self.v_star[p_i])
            #term_1_vec = (self.ps.v[p_j]-self.ps.v[p_i])
            term_2_vec = self.cubic_kernel_derivative(x_i - x_j)
            term_1_mat = ti.Matrix([[term_1_vec[0]], [term_1_vec[1]],
                                    [term_1_vec[2]]])
            term_2_mat = ti.Matrix(
                [[term_2_vec[0], term_2_vec[1], term_2_vec[2]]])
            grad_vel += self.ps.m_V[p_j] * term_1_mat @ term_2_mat

    @ti.func
    def sample_vortex_particle(self, p_i, v_compensation: ti.template()):
        x_i = self.ps.x[p_i]
        for j in range(self.sample_num):
            #if self.neighbor_particle_num[j] >= 20:
            p_j = self.monte_carlo_sample_index[j]
            if p_i == p_j or self.ps.material[p_j] == self.ps.material_solid:
                continue
            x_j = self.ps.x[p_j]
            r_ij = x_i - x_j
            v_compensation += self.delta_vorticity_monte_carlo_samples[
                j].cross(r_ij) / (r_ij.norm()**3)

    @ti.kernel
    def compute_velocity_compensation_monte_carlo(self):
        for p_i in range(self.ps.particle_max_num):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            velocity_compensation = ti.Vector(
                [0.0 for _ in range(self.ps.dim)])
            self.sample_vortex_particle(p_i, velocity_compensation)
            velocity_compensation *= 0.0796 * self.inv_sample_num * self.inv_prob_uni
            if velocity_compensation.norm() < self.ps.v[p_i].norm():
                self.ps.v[p_i] += velocity_compensation
            #self.ps.v[p_i] += velocity_compensation

    @ti.func
    def compute_delta_omega_for_monte_carlo_samples_task(
        self, p_i, p_j, ret: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            x_i = self.ps.x[p_i]
            x_j = self.ps.x[p_j]
            ret.delta_vorticity_montecarlo += self.ps.m_V[
                p_j] * self.delta_vorticity[p_j] * self.cubic_kernel(
                    (x_i - x_j).norm())
            ret.num_neighbors += 1
        elif self.ps.material[p_j] == self.ps.material_solid:
            x_i = self.ps.x[p_i]
            x_j = self.ps.x[p_j]
            ret.delta_vorticity_montecarlo += self.ps.m_V[
                p_j] * self.delta_vorticity[p_j] * self.cubic_kernel(
                    (x_i - x_j).norm())
            ret.num_neighbors += 1

    @ti.func
    def iterate_neighboring_particles(self, p_i, task: ti.template(),
                                      ret: ti.template()):
        x_i = self.ps.x[p_i]
        center_cell = self.ps.pos_to_index(x_i)
        for offset in ti.grouped(ti.ndrange(*((-1, 2), ) * self.ps.dim)):
            grid_index = self.ps.flatten_grid_index(center_cell + offset)
            if grid_index >= 0 and grid_index < self.ps.flattened_grid_num:
                for p_j in range(
                        self.ps.grid_particles_num[ti.max(0, grid_index - 1)],
                        self.ps.grid_particles_num[grid_index]):
                    if p_i != p_j and (x_i - self.ps.x[p_j]
                                       ).norm() < self.ps.support_radius:
                        task(p_i, p_j, ret)

    @ti.kernel
    def compute_delta_omega_for_monte_carlo_samples(self):
        for i in range(self.sample_num):
            p_i = self.monte_carlo_sample_index[i]
            if self.ps.material[p_i] == self.ps.material_solid:
                continue
            self.delta_vorticity_monte_carlo_samples[i] = self.ps.m_V[
                p_i] * self.delta_vorticity[p_i] * self.cubic_kernel(0.0)
            ret = ti.Struct(delta_vorticity_montecarlo=ti.Vector(
                [0.0 for _ in range(self.ps.dim)]),
                            num_neighbors=0)
            self.iterate_neighboring_particles(
                p_i, self.compute_delta_omega_for_monte_carlo_samples_task,
                ret)
            self.delta_vorticity_monte_carlo_samples[
                i] += ret.delta_vorticity_montecarlo
            self.neighbor_particle_num[i] = ret.num_neighbors

    @ti.kernel
    def compute_curl_v(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            curl_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            self.ps.for_all_neighbors(p_i, self.compute_vorticity_task_diff,
                                      curl_v)
            if self.enable_mpsph:
                self.vorticity[p_i] = curl_v

    @ti.kernel
    def convert2flag(
        self, vortex_particles_index_np: ti.types.ndarray()) -> int:
        self.ps.is_sample.fill(0)
        fluid_sample_num = 0
        for i in range(self.sample_num):
            p_i = vortex_particles_index_np[i]
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.is_sample[p_i] = 1
                ti.atomic_add(fluid_sample_num, 1)
        return fluid_sample_num

    @ti.kernel
    def compute_density_change(self):  #compute -nabla dot v
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            ret = ti.Struct(density_adv=0.0, num_neighbors=0)
            self.ps.for_all_neighbors(p_i, self.compute_density_change_task,
                                      ret)

            # only correct positive divergence
            density_adv = ti.max(ret.density_adv, 0.0)
            num_neighbors = ret.num_neighbors

            # Do not perform divergence solve when paritlce deficiency happens
            if self.ps.dim == 3:
                if num_neighbors < 20:
                    density_adv = 0.0
            else:
                if num_neighbors < 7:
                    density_adv = 0.0

            self.ps.density_adv[p_i] = density_adv

    @ti.func
    def compute_density_change_task(self, p_i, p_j, ret: ti.template()):
        v_i = self.ps.v[p_i]
        v_j = self.ps.v[p_j]
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            ret.density_adv += self.ps.m_V[p_j] * (v_i - v_j).dot(
                self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]))
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            ret.density_adv += self.ps.m_V[p_j] * (v_i - v_j).dot(
                self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]))
        # Compute the number of neighbors
        ret.num_neighbors += 1

    #[LIU 2021]
    @ti.kernel
    def vrsph(self):
        for p_i in ti.grouped(self.ps.x):
            sf = ti.Vector([0.0 for _ in range(self.ps.dim)])
            self.ps.for_all_neighbors(p_i, self.compute_streamFunc_task, sf)
            self.streamFunc[p_i] = 0.0796 * sf
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            self.ps.for_all_neighbors(p_i, self.sf2vel_task, d_v)
            d_v *= self.alpha
            if d_v.norm() < self.ps.v[p_i].norm():
                self.ps.v[p_i] += d_v

    @ti.func
    def compute_streamFunc_task(self, p_i, p_j, sf: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            r_ij = self.ps.x[p_i] - self.ps.x[p_j]
            sf += self.ps.m_V[p_j] * self.delta_vorticity[p_j] / (
                r_ij.norm() + 0.1 * self.ps.support_radius)

    @ti.func
    def sf2vel_task(self, p_i, p_j, d_v: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            r_ij = self.ps.x[p_i] - self.ps.x[p_j]
            d_v += self.ps.m_V[p_j] * (self.streamFunc[p_j] -
                                       self.streamFunc[p_i]).cross(
                                           self.cubic_kernel_derivative(r_ij))
        elif self.ps.material[p_j] == self.ps.material_solid:
            r_ij = self.ps.x[p_i] - self.ps.x[p_j]
            d_v += self.ps.m_V[p_j] * (self.streamFunc[p_j] -
                                       self.streamFunc[p_i]).cross(
                                           self.cubic_kernel_derivative(r_ij))

    @ti.kernel
    def record_sample_indices(self):
        index = 0
        self.monte_carlo_sample_index.fill(0)
        for p_i in range(self.ps.particle_max_num):
            if self.ps.is_sample[p_i]:
                index_tmp = ti.atomic_add(index, 1)
                self.monte_carlo_sample_index[index_tmp] = p_i

    def save_data_in_buffer(self):
        self.total_vorticity_buffer.append(self.total_vorticity.to_numpy())
        self.kinetic_energy_buffer.append(self.kinetic_energy.to_numpy())
        self.helicity_buffer.append(self.helicity.to_numpy())
        self.helicity_abs_buffer.append(self.helicity_abs.to_numpy())
        self.rot_energy_buffer.append(self.rot_energy.to_numpy())
        self.potential_energy_buffer.append(self.potential_energy.to_numpy())
        self.total_energy_buffer.append(self.total_energy.to_numpy())
        self.total_energy_plus_rot_energy_buffer.append(
            self.total_energy_plus_rot_energy.to_numpy())
        self.vorticity_norm_sum_buffer.append(
            self.vorticity_norm_sum.to_numpy())

    def divergence_solve(self):
        # TODO: warm start
        # Compute velocity of density change
        self.compute_density_change()
        #inv_dt = 1.0 / self.dt[None]
        self.multiply_time_step(self.ps.dfsph_factor, self.inv_dt)
        m_iterations_v = 0
        # Start solver
        avg_density_err = 0.0
        while m_iterations_v < 1 or m_iterations_v < self.m_max_iterations_v:
            avg_density_err = self.divergence_solver_iteration()
            # Max allowed density fluctuation
            # use max density error divided by time step size
            eta = self.inv_dt * self.max_error_V * 0.01 * self.density_0
            # print("eta ", eta)
            m_iterations_v += 1
            if avg_density_err <= eta:
                break
        #print(f"DFSPH - iteration V: {m_iterations_v} Avg density err: {avg_density_err}")
        # Multiply by h, the time step size has to be removed
        # to make the stiffness value independent
        # of the time step size
        # TODO: if warm start
        # also remove for kappa v
        self.multiply_time_step(self.ps.dfsph_factor, self.dt[None])

    @ti.kernel
    def multiply_time_step(self, field: ti.template(), time_step: float):
        for I in ti.grouped(self.ps.x):
            if self.ps.material[I] == self.ps.material_fluid:
                field[I] *= time_step

    @ti.kernel
    def compute_density_error(self, offset: float) -> float:
        density_error = 0.0
        for I in ti.grouped(self.ps.x):
            if self.ps.material[I] == self.ps.material_fluid:
                density_error += self.ps.density_adv[I] - offset
        return density_error

    @ti.kernel
    def divergence_solver_iteration_kernel(self):
        # Perform Jacobi iteration
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            # evaluate rhs
            b_i = self.ps.density_adv[p_i]
            k_i = b_i * self.ps.dfsph_factor[p_i] * self.ps.density[p_i]
            ret = ti.Struct(dv=ti.Vector([0.0 for _ in range(self.ps.dim)]),
                            k_i=k_i)
            # TODO: if warm start
            # get_kappa_V += k_i
            self.ps.for_all_neighbors(p_i,
                                      self.divergence_solver_iteration_task,
                                      ret)
            self.ps.v[p_i] += ret.dv

    @ti.func
    def divergence_solver_iteration_task(self, p_i, p_j, ret: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            b_j = self.ps.density_adv[p_j]
            k_j = b_j * self.ps.dfsph_factor[p_j] * self.ps.density[p_j]
            #k_sum = ret.k_i + self.density_0 / self.density_0 * k_j  # TODO: make the neighbor density0 different for multiphase fluid
            k_sum = ret.k_i + k_j
            if ti.abs(k_sum) > self.m_eps:
                grad_p_j = self.ps.m[p_j] * self.cubic_kernel_derivative(
                    self.ps.x[p_i] - self.ps.x[p_j])
                ret.dv -= self.dt[None] * k_sum * grad_p_j
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            if ti.abs(ret.k_i) > self.m_eps:
                grad_p_j = self.ps.m_V[
                    p_j] * self.density_0 * self.cubic_kernel_derivative(
                        self.ps.x[p_i] - self.ps.x[p_j])
                vel_change = -self.dt[None] * 1.0 * ret.k_i * grad_p_j
                ret.dv += vel_change
                if self.ps.is_dynamic_rigid_body(p_j):
                    self.ps.acceleration[
                        p_j] += -vel_change * self.inv_dt * self.ps.density[
                            p_i] / self.ps.density[p_j]

    @ti.kernel
    def compute_density_adv(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            delta = 0.0
            self.ps.for_all_neighbors(p_i, self.compute_density_adv_task,
                                      delta)
            density_adv = self.ps.density[p_i] + self.dt[None] * delta
            self.ps.density_adv[p_i] = ti.max(density_adv, self.density_0)

    @ti.func
    def compute_density_adv_task(self, p_i, p_j, ret: ti.template()):
        v_i = self.ps.v[p_i]
        v_j = self.ps.v[p_j]
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            ret += self.ps.m[p_j] * (v_i - v_j).dot(
                self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]))
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            ret += self.ps.m_V[p_j] * self.density_0 * (v_i - v_j).dot(
                self.cubic_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]))

    @ti.kernel
    def pressure_solve_iteration_kernel(self):
        # Compute pressure forces
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            # Evaluate rhs
            b_i = self.ps.density_adv[p_i] - self.density_0
            k_i = b_i * self.ps.dfsph_factor[p_i]

            # TODO: if warmstart
            # get kappa V
            self.ps.for_all_neighbors(p_i, self.pressure_solve_iteration_task,
                                      k_i)

    @ti.func
    def pressure_solve_iteration_task(self, p_i, p_j, k_i: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            b_j = self.ps.density_adv[p_j] - self.density_0
            k_j = b_j * self.ps.dfsph_factor[p_j]
            #k_sum = k_i + self.density_0 / self.density_0 * k_j # TODO: make the neighbor density0 different for multiphase fluid
            k_sum = k_i + k_j
            if ti.abs(k_sum) > self.m_eps:
                grad_p_j = self.ps.m[p_j] * self.cubic_kernel_derivative(
                    self.ps.x[p_i] - self.ps.x[p_j])
                # Directly update velocities instead of storing pressure accelerations
                self.ps.v[p_i] -= self.dt[
                    None] * k_sum * grad_p_j  # ki, kj already contain inverse density
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            if ti.abs(k_i) > self.m_eps:
                grad_p_j = self.ps.m_V[
                    p_j] * self.density_0 * self.cubic_kernel_derivative(
                        self.ps.x[p_i] - self.ps.x[p_j])

                # Directly update velocities instead of storing pressure accelerations
                vel_change = -self.dt[
                    None] * 1.0 * k_i * grad_p_j  # kj already contains inverse density
                self.ps.v[p_i] += vel_change
                if self.ps.is_dynamic_rigid_body(p_j):
                    self.ps.acceleration[
                        p_j] += -vel_change * self.inv_dt * self.ps.density[
                            p_i] / self.ps.density[p_j]

    @ti.kernel
    def update_in_dynamic_area_tag(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_in_dynamic_area[
                    p_i] == False and self.ps.x[p_i][0] > 3:
                self.ps.is_in_dynamic_area[p_i] = True

    @ti.kernel
    def rotate_propeller(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[
                    p_i] != self.ps.material_solid or self.ps.object_id[
                        p_i] != 1:
                continue
            pos = self.ps.x[p_i]
            pos -= self.offset_propeller
            pos = self.propeller_rot_mat[None] @ pos
            pos += self.offset_propeller
            self.ps.x[p_i] = pos

    @ti.kernel
    def restrict_upward(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            if self.ps.x[p_i][1] > 0.8 and self.ps.v[p_i][1] > 0:
                self.ps.v[p_i][1] *= ti.exp(-0.01 * self.ps.v[p_i][1])

    def pressure_solve_iteration(self):
        self.pressure_solve_iteration_kernel()
        self.compute_density_adv()
        density_err = self.compute_density_error(self.density_0)
        return density_err / self.ps.fluid_particle_num

    def divergence_solver_iteration(self):
        self.divergence_solver_iteration_kernel()
        self.compute_density_change()
        density_err = self.compute_density_error(0.0)
        return density_err / self.ps.fluid_particle_num

    def pressure_solve(self):
        inv_dt2 = self.inv_dt2

        # TODO: warm start
        # Compute rho_adv
        self.compute_density_adv()

        self.multiply_time_step(self.ps.dfsph_factor, inv_dt2)

        m_iterations = 0

        # Start solver
        avg_density_err = 0.0

        while m_iterations < 1 or m_iterations < self.m_max_iterations:
            avg_density_err = self.pressure_solve_iteration()
            # Max allowed density fluctuation
            eta = self.max_error * 0.01 * self.density_0
            m_iterations += 1
            if avg_density_err <= eta:
                break
        #print(f"DFSPH - iterations: {m_iterations} Avg density Err: {avg_density_err:.4f}")

    def generate_monte_carlo_samples(self):
        random_particle_index_array = np.clip(
            (np.random.rand(self.sample_num) *
             self.ps.particle_max_num).astype(int), 0,
            self.ps.particle_max_num - 1)
        fluid_sample_num = self.convert2flag(random_particle_index_array)
        return fluid_sample_num

    def vortex_particle_monte_carlo(self):
        self.record_sample_indices()
        self.compute_delta_omega_for_monte_carlo_samples()
        self.compute_velocity_compensation_monte_carlo()

    def substep(self):
        self.compute_densities()
        self.compute_DFSPH_factor()
        self.divergence_solve()
        self.compute_non_pressure_forces()
        if self.enable_mpsph:
            self.micropolar_solve()
        elif self.enable_vrsph:
            self.compute_delta_vorticity_for_sph_particles()
            self.vrsph()
        elif self.enable_mcvsph:
            self.compute_delta_vorticity_for_sph_particles()
            self.vortex_particle_monte_carlo()
        self.predict_velocity()
        self.pressure_solve()
        self.advect()
        if self.show_curl_color:
            self.compute_particles_color_curl()

        if self.is_propeller:
            self.rotate_propeller()
            self.restrict_upward()
