import u_omega.sop_module as sop_m
import u_omega.uom_module as uom_m
import u_omega.data_module as data_m
import u_omega.util_module as util_m
import u_omega.qepp_module as qepp_m
import u_omega.aim_module as aim_m

import u_omega_old.UOM_pkg.locGWnscf as locgw_m
import u_omega_old.UOM_pkg.fromXML_to_JSONband as xml2json_m
import u_omega_old.UOM_pkg.uomAI as uom_old_m

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json

reset = True
run_qe = False

if run_qe == False: reset = False
if reset:
    os.system("rm -rf output")
    os.system("rm -rf results")
    os.system("rm -rf projwfc")

nsteps = 10

wgrid = np.linspace(-100., 100., 1000)

def do_step(data: data_m.Data, i=0, smearing_w_old=None):

    fig, ax = plt.subplots()

    print("Building kInfoScf...")

    atomDictList = [ 
          {
              "atom"       : "V",
              "atomIndex"  :  1,
              "indexList"  :  [5,6,7,8,9], 
              "lIndex"     :  2,
              "localize"   :  True
          }
      ]

    kInfoScf = xml2json_m.get('./results/SrVO3.save/atomic_proj.xml', './results/SrVO3.save/data-file-schema.xml', 'SrVO3.json', atomDictList, dmatFileSuffix='sym.dat')

    print("Building h0 and g0...")
    h0 = qepp_m.build_ham_ks(data)
    g0 = qepp_m.build_g0ks_from_ham(h0, data)
    g0_sm = sop_m.smear(g0, data, data.efermi_ks0)
    g0loc = uom_m.ks_2_loc(g0, data)
    print(g0loc[0].npoles)
    # g0loc = sop_m.condense(g0loc, 1.e-6, sop_m.weighted_average, sop_m.same_thr)
    g0loc = sop_m.cond_and_epurate(g0loc, 1.e-2, 1.e-8)

    G0Ks = uom_old_m.A0SOPListG0KsFromHKs(h0)
    mu, G0Ks_sm = locgw_m.findMuAndsmearG(G0Ks, data.w_k, data.n_electrons, sop_m.marzari_vanderbilt_smearing, data.degauss, smearing_w_old, data.efermi_ks0)

    G0Ks_sm = sop_m.from_json_mat_2_sop(G0Ks_sm[0])

    g0loc_old = locgw_m.localizeGList(G0Ks, data.w_k, data.u_mats, 1, nonCol=False, dMat=data.d_mats[2], irt=data.irt, atomIndexList=[1])
    g0loc_old = [sop_m.from_json_mat_2_sop(g0loc_old[iatom]) for iatom in range(data.n_atoms)]
    g0loc_old = sop_m.cond_and_epurate(g0loc_old, 1.e-2, 1.e-8)
    for iatom in range(data.n_atoms):
        g0loc_old[iatom].efermi = g0loc[iatom].efermi

    # plot before reconverting
    g0loc_old[0].plot(ax, wgrid, True, "g0loc_old", color=["cyan", "magenta"], lw=4)
    g0loc[0].plot(ax, wgrid, True, "g0loc", color=["blue", "red"], lw=2)
    plt.show()
    plt.close()
    print(np.max(np.abs(g0loc[0].evaluate(wgrid, True)-g0loc_old[0].evaluate(wgrid, True))))

    # reconvert
    g0loc_old = [sop_m.from_sop_2_json(g0loc_old[iatom]) for iatom in range(data.n_atoms)]

    print("Building self-energy...")
    self_energy = uom_m.compute_loc_self_en(g0loc, data)
    # self_energy = sop_m.cond_and_epurate(self_energy, 1.e-0, 1.e-3)

    self_energy_old, occ_loc_list, e_hub_old = locgw_m.sigmaLocAndPhiList(g0loc_old, sop_m.from_sop_2_json(data.u_sop[0]), 1)

    self_energy_old = [sop_m.from_json_2_sop(self_energy_old[iatom]) for iatom in range(data.n_atoms)]
    # self_energy_old = sop_m.cond_and_epurate(self_energy_old, 1.e-0, 1.e-3)

    for iatom in range(data.n_atoms):
        self_energy_old[iatom].efermi = self_energy[iatom].efermi
    print(np.max(np.abs(self_energy[0].evaluate(wgrid, True)-self_energy_old[0].evaluate(wgrid, True))))
    #
    fig, ax = plt.subplots()
    self_energy_old[0].plot(ax, wgrid, True, "g0loc_old", color=["cyan", "magenta"], lw=4)
    self_energy[0].plot(ax, wgrid, True, "g0loc", color=["blue", "red"], lw=2)
    plt.show()
    plt.close()

    ehub = uom_m.compute_hub_energy(g0loc, self_energy, data)
    print("ehub", ehub)
    print("ehub_old", e_hub_old)
    exit()

    self_energy_old = [sop_m.from_sop_2_json(self_energy_old[iatom]) for iatom in range(data.n_atoms)]

    print("Building g1...")
    g1_old = locgw_m.GKSfromSigmaLocList(self_energy_old, h0, data.u_mats, 1, 0, 0.)[0]
    g1 = aim_m.dyson_aim_ks(h0, self_energy, data)

    g1_old = [sop_m.from_json_mat_2_sop(g1_old[ik]) for ik in range(data.nk)]
    g1_old = sop_m.cond_and_epurate(g1_old, 1.e-2, 1.e-8)
    sop_m.write(g1_old, "results/g1_ik0_old.sop")
    g1_old = [sop_m.from_sop_2_json(g1_old[ik]) for ik in range(data.nk)]

    g1 = sop_m.cond_and_epurate(g1, 1.e-2, 1.e-8)
    sop_m.write(g1[0], "results/g1_ik0.sop")

    g1loc = uom_m.ks_2_loc(g1, data)
    g1loc = sop_m.cond_and_epurate(g1loc, 1.e-2, 1.e-8)
    g1loc_old = locgw_m.localizeGList(g1_old, data.w_k, data.u_mats, 1, nonCol=False, dMat=data.d_mats[2], irt=data.irt, atomIndexList=[1])
    g1loc_old = [sop_m.from_json_mat_2_sop(g1loc_old[iatom]) for iatom in range(data.n_atoms)]
    g1loc_old = sop_m.cond_and_epurate(g1loc_old, 1.e-2, 1.e-8)
    for iatom in range(data.n_atoms):
        g1loc_old[iatom].efermi = g1loc[iatom].efermi

    fig, ax = plt.subplots()
    g1loc_old[0].plot(ax, wgrid, True, "g1loc_old", color=["cyan", "magenta"], lw=4)
    g1loc[0].plot(ax, wgrid, True, "g1loc", color=["blue", "red"], lw=2)
    plt.show()
    fig.savefig(f"results/g1loc_comparison_istep{i}.png")
    plt.close()



    occ_mat = []
    occ_mat_old = uom_old_m.A0SOPListOccupiedNMoment(g1_old, 0)
    for ik in range(data.nk):
        occ_mat.append(g1[ik].occupied_moment(0))

    occ_mat = np.array(occ_mat)
    occ_mat_old = np.array(occ_mat_old)
    print(np.max(np.abs(occ_mat-occ_mat_old)))

    locgw_m.writeDensityMat(occ_mat_old, "densityMat.dat")

    pass

if os.path.isdir("output") == False: os.mkdir("output")
if os.path.isdir("results") == False: os.mkdir("results")
if os.path.isdir("projwfc") == False: os.mkdir("projwfc")

if run_qe:
    qe_path = "/home/pintus/Documents/codes/q-e/build/bin/"
    mpi_comm = "mpirun -np 12 "

    os.system(mpi_comm+qe_path+"pw.x <  input/SrVO3.scf.in >  output/SrVO3.scf.out")

    os.system(mpi_comm+qe_path+"projwfc.x < input/SrVO3.projwfc.in > output/SrVO3.projwfc.out")

    os.system(mpi_comm+qe_path+"chargedens_fromfile.x < input/SrVO3.symmetries.in > output/SrVO3.symmetries.out")

data = data_m.Data()
input_dict = "input/input_dict.json"
data.read_input_json(input_dict)
smearing_w_old = {"LB": data.smearing_w[0], "RB": data.smearing_w[1]}
i = -1
data.get_qe_data("sym.dat")
do_step(data, i, smearing_w_old)
# exit()

for i in range(nsteps):

    qe_path = "/home/pintus/Documents/codes/q-e/build/bin/"
    mpi_comm = "mpirun -np 12 "

    os.system(mpi_comm+qe_path+"pw.x <  input/SrVO3.scf.in >  output/SrVO3.scf.out")

    os.system(mpi_comm+qe_path+"projwfc.x < input/SrVO3.projwfc.in > output/SrVO3.projwfc.out")

    os.system(mpi_comm+qe_path+"chargedens_fromfile.x < input/SrVO3.symmetries.in > output/SrVO3.symmetries.out")

    data.get_qe_data("sym.dat")

    do_step(data, i)

    os.system(mpi_comm+qe_path+"chargedens_fromfile.x < input/SrVO3.chargedens.in > output/SrVO3.chargedens.out")
