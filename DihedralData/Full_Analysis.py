from os import listdir, system, mkdir
from os.path import join, exists, abspath, dirname
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import MDAnalysis as MDA
from calcs import *
import pyDIFRATE as DR
from numba.cuda import is_available as cuda_available

# setting default parameters for matplotlib plots
matplotlib.rcParams.update({"lines.linewidth": 1,
                            "axes.labelsize": 8,
                            "xtick.labelsize": 8,
                            "ytick.labelsize": 8,
                            "axes.titlesize": 10,
                            'font.size': 8})
TEXTBOX = {'facecolor': 'lightblue',
           'alpha': 0.5}
TB2 = {'facecolor': 'red',
           'alpha': 0.7}

from SpeedTest import time_runtime

class KaiMarkov():
    def __init__(self, **kwargs):
        self.n = 50
        self.offset = 147
        self.length = None
        self.labels = []
        self.full_chains = []
        self.plot_hydrogens_rama_3D = []  # get a better name for that, for ramachandran 3d plot
        self.hydrogens_dihedral = []
        self.chi1_groups_dihedral = []
        self.chi2_groups_dihedral = []
        self.ct_vector_groups = []  # Atomgroups to get coordinates for calculation of correlation function
        self.n_dets = None  # number of detectors for analysis
        self.part = None  # the number of frames that will be in one fraction defined by length/n
        self.residues = []  # residues to examine, can be preset in kwargs
        self.pdb = None  # pdb file (path) that will fit to the selected simulation
        self.sel_sim = None  # filename of selected simulation
        self.sel_sim_name = None  # same just with removed xtc
        self.universe = None  # will be MDA.Universe object
        self.default_float = "float32"  # used for all kinds of calculations and saves some space compared to float 64
        self.save_float = "float16"  # saves space for saving issues, since md accuracy is only 0.01 it shouldn't matter
        self.v = False  # verbose, if activated, print
        self.full_dict = {}
        self.dir = dirname(__file__)
        if not exists(join(self.dir, "cts")):
            mkdir(join(self.dir, "cts"))
        for key in kwargs:
            setattr(self, key, kwargs[key])
        if self.sel_sim is None:
            self.select_simulation(kwargs.get("simulation"))
        self.get_methyl_groups()

        # the length of self.dih is 3 times the number of methylgroups, as well as the length of areas and hops
        self.dih = np.zeros((len(self.hydrogens_dihedral), self.length), dtype=self.default_float)
        self.dih_chi1 = np.zeros((len(self.chi1_groups_dihedral), self.length), dtype=self.default_float)
        self.dih_chi2 = np.zeros((len(self.chi2_groups_dihedral), self.length), dtype=self.default_float)

        self.areas = np.zeros((len(self.hydrogens_dihedral), self.length), dtype="uint8")
        self.chi1_areas = np.zeros((len(self.chi1_groups_dihedral), self.length), dtype="uint8")
        self.chi2_areas = np.zeros((len(self.chi2_groups_dihedral), self.length), dtype="uint8")
        self.coords_3Dplot = np.zeros((len(self.hydrogens_dihedral), 2000, 3), dtype=self.default_float)
        # creating arrays for all occuring hops over the trajectory and for the hop probabilities ocuring in single
        # fragments (n)
        self.hops = np.zeros((len(self.hydrogens_dihedral), self.length), dtype=bool)
        self.hopability = np.zeros((len(self.hydrogens_dihedral), self.n), dtype=self.default_float)
        self.hops_chi1 = np.zeros((len(self.chi1_groups_dihedral), self.length), dtype=bool)
        self.hopability_chi1 = np.zeros((len(self.chi1_groups_dihedral), self.n), dtype=self.default_float)
        self.hops_chi2 = np.zeros((len(self.chi2_groups_dihedral), self.length), dtype=bool)
        self.hopability_chi2 = np.zeros((len(self.chi2_groups_dihedral), self.n), dtype=self.default_float)

        self.ct_vectors = np.zeros((len(self.ct_vector_groups), self.length, 3), dtype=self.default_float)
        self.cts = np.ones((len(self.ct_vectors), self.length), dtype=self.default_float)
        self.S2s = np.zeros((len(self.ct_vectors)), self.default_float)

    def get_methyl_groups(self):
        def add_ct(label, atoms):
            this["ct_labels"].append(label)
            self.ct_vector_groups.append(MDA.AtomGroup([this[_] for _ in atoms]))
            this["ct_indices"].append(len(self.ct_vector_groups) - 1)

        assert self.sel_sim and self.pdb, "You have to select a PDB file AND a simulation before "
        if self.universe is None:
            self.universe = MDA.Universe(self.pdb, self.sel_sim)
        uni = self.universe  # just to shorten this
        # TODO this will just work for HETs right now, make it more general
        seg = 1 if len(uni.segments) == 3 else 2 if len(uni.segments) == 5 else 0
        # TODO add segment selection to select simulation function
        segment = uni.segments[seg]
        residues = [segment.residues[r] for r in self.residues] if len(self.residues) else segment.residues
        dix = self.full_dict
        for i, res in enumerate(residues):
            methyls = search_methyl_groups(res, v=self.v)
            if len(methyls):
                group_label = "{}_{}".format(res.resname, res.resnum + self.offset)
                this = dix[group_label] = {}

                this["ct_indices"] = []
                this["ct_labels"] = []
                print(i, res.resname, res.resnum + self.offset, "contains", len(methyls), "methyl groups")
                #self.residues.append(i)
                chain_for_3d_plot = dix[group_label]["chain"] = ["C", "CA", "N", "CA", "CB"]
                this["N"] = res.atoms[0]
                this["H1"] = methyls[0][0]  # this is only useful if you have only one H
                this["H12"] = methyls[0][1]
                this["H13"] = methyls[0][2]
                this["C"] = methyls[0][-1]
                this["CA"] = methyls[0][-2]
                this["CB"] = methyls[0][-3]
                chi1 = chi2 = 0
                add_ct(r'$\chi_{1,lib.}$' if "ALA" not in res.resname else r'$CH_{3,lib.}$', ["CB", "N", "CA", "C"])

                # todo add methionin
                if "ILE" in res.resname:
                    for _ in ["CG2", "CB", "CG1", "CD"]:
                        chain_for_3d_plot.append(_)
                    this["CG2"] = methyls[0][-4]
                    this["CG1"] = methyls[1][4]
                    this["CD"] = methyls[1][3]
                    this["H2"] = methyls[1][0]
                    #add_ct("CB_CG2", ["CG2", "C", "CB", "CA"])  # this is also Chi1 rot basically
                    add_ct(r'$\chi_{1,rot.}$', ["CG1", "C", "CB", "CA"])  # "CB_CG1"
                    add_ct(r'$\chi_{2,lib.}$', ["CG1", "CA", "CB", "CG2"])
                    add_ct(r'$\chi_{2,rot.}$', ["CD", "CB", "CG1", "CG2"])  # "CG1_CD"
                    add_ct(r'$CH_{3,rot.}^1$', ["H1", "CB", "CG2", "CG1"])  # CG2_H1
                    add_ct(r'$CH_{3,rot.}^2$', ["H2", "CG1", "CD", "CB"])  # CD_H2
                    #TODO maybe uncomment this sometime, will give you total movement of the CH vector compared to the
                    #TODO peptide plane
                    #add_ct(r'$CH_{3,tot.}^1$',["H1","N","CA","C","CG2"])
                    #add_ct(r'$CH_{3,tot.}^2$', ["H2", "N", "CA", "C", "CD"])
                    self.chi1_groups_dihedral.append(MDA.AtomGroup([this["C"], this["CA"], this["CB"], this["CG1"]]))
                    self.chi2_groups_dihedral.append(MDA.AtomGroup([this["CA"], this["CB"], this["CG1"], this["CD"]]))
                    chi1 = chi2 = 1
                elif "VAL" in res.resname:
                    for _ in ["CG1", "CB", "CG2"]:
                        chain_for_3d_plot.append(_)
                    this["CG1"] = methyls[0][-4]  # in "VAL" this is CG1
                    this["CG2"] = methyls[1][3]  # in "VAL this is CG2
                    this["H2"] = methyls[1][0]
                    add_ct(r'$\chi_{1,rot.}$', ["CG1", "C", "CB", "CA"])  # "CB_CG1"
                    add_ct(r'$CH_{3,lib.}^2$', ["CG2", "CG1", "CB", "CA"])  # "CB_CG2"
                    add_ct(r'$CH_{3,rot.}^1$', ["H1", "CB", "CG1", "CG2"])
                    add_ct(r'$CH_{3,rot.}^2$', ["H2", "CB", "CG2", "CG1"])
                    #add_ct(r'$CH_{3,tot.}^1$',["H1","N","CA","C","CG1"])
                    #add_ct(r'$CH_{3,tot.}^2$', ["H2", "N", "CA", "C", "CG2"])
                    self.chi1_groups_dihedral.append(MDA.AtomGroup([this["C"], this["CA"], this["CB"], this["CG2"]]))
                    chi1 = 1
                elif "LEU" in res.resname:
                    for _ in ["CG", "CD1", "CG", "CD2"]:
                        chain_for_3d_plot.append(_)
                    this["CG"] = methyls[0][4]
                    this["CD1"] = methyls[0][3]
                    this["CD2"] = methyls[1][3]
                    this["H2"] = methyls[1][0]
                    add_ct(r'$\chi_{1,rot.}$', ["CG", "CA", "CB", "C"])  # "CB_CG"
                    add_ct(r'$\chi_{2,rot.}$', ["CD1", "CA", "CG", "CB"])  # "CG_CD1"
                    # add_ct("CG_CD2", ["CD2", "CA", "CG", "CB"])
                    add_ct(r'$CH_{3,rot.}^1$', ["H1", "CG", "CD1", "CD2"])  # "CD1_H1"
                    add_ct(r'$CH_{3,rot.}^2$', ["H2", "CG", "CD2", "CD1"])  # "CD2_H2"
                    #add_ct(r'$CH_{3,tot.}^1$',["H1","N","CA","C","CD1"])
                    #add_ct(r'$CH_{3,tot.}^2$', ["H2", "N", "CA", "C", "CD2"])
                    self.chi1_groups_dihedral.append(MDA.AtomGroup([this["C"], this["CA"], this["CB"], this["CG"]]))
                    self.chi2_groups_dihedral.append(MDA.AtomGroup([this["CA"], this["CB"], this["CG"], this["CD2"]]))
                    chi1 = chi2 = 1
                elif "THR" in res.resname:
                    this["CG2"] = methyls[0][3]
                    chain_for_3d_plot.append("CG2")
                    add_ct(r'$\chi_{1,rot.}$', ["CG2", "CA", "CB", "C"])  # "CB_CG2"
                    add_ct(r'$CH_{3,rot.}$', ["H1", "CA", "CG2", "CB"])  # "CG2_H1"
                    #add_ct(r'$CH_{3,tot.}^1$', ["H1", "N", "CA", "C", "CG2"])
                    self.chi1_groups_dihedral.append(MDA.AtomGroup([this["C"], this["CA"], this["CB"], this["CG2"]]))
                    chi1 = 1
                elif "ALA" in res.resname:
                    add_ct(r'$CH_{3,rot.}$', ["H1", "CA", "CB", "C"])  # "CB_H1"
                    add_ct(r'$C-H_{lib.}$', ["H1", "H12", "CB", "CA"])
                    #add_ct(r'$CH_{3,tot.}^1$', ["H1", "N", "CA", "C", "CB"])
                    # add_ct(r'$C-H_{lib.}H3$',["H1", "H12", "CB", "H13"])
                if chi1: this['chi1'] = len(self.chi1_groups_dihedral) - 1
                if chi2: this['chi2'] = len(self.chi2_groups_dihedral) - 1
                this["labels"] = []
                for j, m in enumerate(methyls):
                    met_label = group_label + ("_B" if j else ("_A" if len(methyls) == 2 else ""))
                    this["labels"].append(met_label)
                    this[met_label] = {}
                    sdix = this[met_label]  # subdictionary
                    sdix["index"] = len(self.hydrogens_dihedral)
                    sdix["carbon"] = m[3]
                    group = MDA.AtomGroup(m)
                    self.full_chains.append(group)
                    self.plot_hydrogens_rama_3D.append(group[[-1, -2, -3, 0]])
                    self.plot_hydrogens_rama_3D.append(group[[-1, -2, -3, 1]])
                    self.plot_hydrogens_rama_3D.append(group[[-1, -2, -3, 2]])
                    # One group appended for every hydrogen
                    # it is a little weird but still this was the fastest way to calculate the dih for this
                    self.hydrogens_dihedral.append(
                        group[[0, 3, 4, 5]])  # presetting this saves much time for calculations
                    self.hydrogens_dihedral.append(group[[1, 3, 4, 5]])
                    self.hydrogens_dihedral.append(group[[2, 3, 4, 5]])

        assert len(self.hydrogens_dihedral), "No methylgroups detected"
        print("Total:", int(len(self.hydrogens_dihedral) / 3), "methylgroups detected")
        if self.v:
            for key in dix.keys():
                print(key)
                print(dix[key])

    def select_simulation(self, number=None):
        xtcdir = join(self.dir, "xtcs")
        pdbdir = join(self.dir, "pdbs")
        if not exists(xtcdir) or not exists(pdbdir):
            mkdir(xtcdir)
            mkdir(pdbdir)
            assert 0, "folders for xtcs and pdbs were created, plz fill with trajectories and pdbfiles"
        simulations = [f for f in listdir(xtcdir) if f.endswith(".xtc")]
        assert len(simulations), "please copy (or link) some simulations in your xtcs/ folder"
        assert len(listdir(pdbdir)), "of course you also need a pdb file (in pdbs/)"
        if number is None:
            print("Available Simulations:")
            for i, sim in enumerate(simulations):
                print(i, sim)
            i = int(input("Choosen: "))
        else:
            i = number
        try:
            self.sel_sim = join(xtcdir, simulations[i])
            self.sel_sim_name = simulations[i].split(".")[0]
        except:
            print("not a valid simulation")
            self.select_simulation()

        for pdb in listdir(pdbdir):
            f = join(pdbdir, pdb)
            print(f)
            try:
                self.universe = MDA.Universe(f, self.sel_sim)
                self.pdb = f
                break
            except:
                self.universe = None

        assert self.universe, "Couldnt find a fitting pdb for this simulation, please copy one in the pdb folder"
        print("Simulation", self.sel_sim_name, "with timestep", self.universe.trajectory.dt, "and", self.pdb,
              "selected")
        self.dt = self.universe.trajectory.dt * 10 ** (-12)  # 5 ps
        if self.length == None:
            self.length = len(self.universe.trajectory)
        # todo put segment selection here if more than one segment is available

    def save_state(self):
        dir = join(self.dir,"calced_data")
        if not exists(dir):
            mkdir(dir)
        fn = join(dir, "{}_{}.npy")
        if len(self.residues) == 0 and len(self.universe.trajectory) == self.length:
            np.save(fn.format("dih",self.sel_sim_name),self.dih.astype(self.save_float))
            np.save(fn.format("dih_chi1", self.sel_sim_name), self.dih_chi1.astype(self.save_float))
            np.save(fn.format("dih_chi2", self.sel_sim_name), self.dih_chi2.astype(self.save_float))
            np.save(fn.format("cts", self.sel_sim_name), self.cts.astype(self.save_float))
            np.save(fn.format("S2s", self.sel_sim_name), self.S2s.astype(self.save_float))
            print("dihedrals, cts and S2s saved into file")


    def load_state(self):
        fn = join(self.dir,"calced_data","{}_{}.npy")
        if exists(fn.format("dih",self.sel_sim_name)) and len(self.residues)==0 and len(self.universe.trajectory)==self.length:
            dih = np.load(fn.format("dih",self.sel_sim_name)).astype(self.default_float)
            assert dih.shape[0]==self.dih.shape[0], "number of requested dihedrals not fitting"
            self.dih = dih
            self.dih_chi1 = np.load(fn.format("dih_chi1",self.sel_sim_name)).astype(self.default_float)
            self.dih_chi2 = np.load(fn.format("dih_chi2",self.sel_sim_name)).astype(self.default_float)
            cts = np.load(fn.format("cts",self.sel_sim_name)).astype(self.default_float)
            assert cts.shape[0]==self.cts.shape[0], "Number of requested cts not fitting, {} vs {}\n" \
                                                    "Maybe you added/removed a ct in self.get_methyl_groups. If yes, " \
                                                    "undo it or manually remove files in calced_data/" \
                                                    "".format(cts.shape[0],self.cts.shape[0])
            self.cts = cts
            self.S2s = np.load(fn.format("S2s",self.sel_sim_name)).astype(self.default_float)
            print("dihedrals, cts and S2s loaded from file")
            return 1
        else:
            return 0

    @time_runtime
    def calc(self, **kwargs):
        '''doing all importand calculations for the anylsis
        available kwargs:
        "ct_off" disabeling the calculation of ct, for time saving purposes
        "recalc_ct" forcing to recalculate cts even if they are already stored on drive
        "sparse" for ct calculation, low sparse value decreases computation time and increases noise'''
        traj = self.universe.trajectory
        assert self.length <= len(traj), "Length {} exceeding length of trajectory ({}), set length to {}" \
                                         "".format(self.length, len(traj), len(traj))
        if not self.load_state() or kwargs.get("recalc"):
            # iterating over all timepoints defined by length and calculating dihedrals for all methyl hydrogens,
            # chi_1 and chi_2 bonds. Furthermore calculating the vertices for later C(t) calculation
            t = time()
            for i in range(self.length):
                if i % 10000 == 0:
                    print(i,"von",self.length,time()-t)
                    t=time()
                traj[i]
                for j, group in enumerate(self.hydrogens_dihedral):  # first hydrogen
                    self.dih[j, i] = fastest_dihedral(group.positions)
                for j, group in enumerate(self.chi1_groups_dihedral):
                    self.dih_chi1[j, i] = fastest_dihedral(group.positions)
                for j, group in enumerate(self.chi2_groups_dihedral):
                    self.dih_chi2[j, i] = fastest_dihedral(group.positions)
                #for j, group in enumerate(self.ct_vector_groups):
                #    pos_xyz_o(self.ct_vectors[j, i],*group.positions)
            print("Warning, you excluded calculation of the cts!")
            if cuda_available():
                for i in range(self.S2s.shape[0]//10+1):
                    indices = np.zeros(self.S2s.shape)
                    indices[i*10:(i+1)*10] = 1
                    #calc_CT_on_cuda(self.cts, self.S2s, self.ct_vectors, indices.astype(bool),
                    #            kwargs.get("sparse") if kwargs.get("sparse") is not None else 1)
            #else:
            #    get_ct_S2(self.cts, self.S2s, self.ct_vectors,
            #          kwargs.get("sparse") if kwargs.get("sparse") is not None else 1)
            #if kwargs.get("sparse")==0:
            #    self.save_state()
        self.cts[:,0] = 1
        for i, _ in enumerate(range(0, self.length, int(self.length/2000))):
            # todo calculate a good number of points for the plot
            if _ % 10000 == 0:                print(_)
            traj[_]
            if i == 2000:
                break
            for j, group in enumerate(self.plot_hydrogens_rama_3D):  # first hydrogen
                self.coords_3Dplot[j, i] = get_x_y_z(group.positions)
        # calculating in which of three areas a methyl hydrogen appears at every timepoint
        # furthermore check if a hop between timepoints occured by comparing the areas
        self.areas[:, :] += ((self.dih[:, :] >= 0) == (self.dih[:, :] < 120)).astype('uint8')
        self.areas[:, :] += ((self.dih[:, :] < 0) == (self.dih[:, :] >= -120)).astype('uint8') * 2
        self.hops[:, 1:] = self.areas[:, :-1] != self.areas[:, 1:]  # the first element of hops is by definition 0
        # same for chi1
        self.chi1_areas[:, :] += ((self.dih_chi1[:, :] >= 0) == (self.dih_chi1[:, :] < 120)).astype('uint8')
        self.chi1_areas[:, :] += ((self.dih_chi1[:, :] < 0) == (self.dih_chi1[:, :] >= -120)).astype('uint8') * 2
        self.hops_chi1[:, 1:] = self.chi1_areas[:, :-1] != self.chi1_areas[:, 1:]
        # and chi2
        self.chi2_areas[:, :] += ((self.dih_chi2[:, :] >= 0) == (self.dih_chi2[:, :] < 120)).astype('uint8')
        self.chi2_areas[:, :] += ((self.dih_chi2[:, :] < 0) == (self.dih_chi2[:, :] >= -120)).astype('uint8') * 2
        self.hops_chi2[:, 1:] = self.chi2_areas[:, :-1] != self.chi2_areas[:, 1:]
        # how often does it happen that 2 hydrogens appear in the same area?
        #print(np.sum(self.areas[0::3, :] == self.areas[1::3, :]))  # TODO i dont know what todo with it
        #print(np.sum(self.areas[0::3, :] == self.areas[2::3, :]))  # TODO
        #print(np.sum(self.areas[1::3, :] == self.areas[2::3, :]))  # TODO
        self.part = part = int(self.length / self.n)
        # calculating the hop probabilty for every methylgroup and chi1 and chi2 bond for every fraction n
        for i in range(self.n):
            for j in range(0, len(self.hydrogens_dihedral), 3):
                self.hopability[j, i] = np.sum(self.hops[j, i * part:(i + 1) * part]) / part
            for j in range(len(self.chi1_groups_dihedral)):
                self.hopability_chi1[j, i] = np.sum(self.hops_chi1[j, i * part:(i + 1) * part]) / part
            for j in range(len(self.chi2_groups_dihedral)):
                self.hopability_chi2[j, i] = np.sum(self.hops_chi2[j, i * part:(i + 1) * part]) / part

    @time_runtime
    def plot_all(self):
        """this function creates the plots for all examined residues depending on the number of methyl groups and
        if a chi1 and/or chi2 bond is contained. All plots will be saved in a predefined folder (now test/). If only
        a single residue is examined the plot will be immediately shown (but still saved)"""
        def plot_hopability_with_chi_states(axis, index, meth_type, chi_num, chi_type, hoptype="methyl"):
            """there wasnt a very smooth way to put this calculation in the self.calc function, so i put it here
            if a chi2 bond is existing, it will separate the hop probability in three bars to show in which state of
            chi2 the hops are occuring. furthermore, if a shift_axis is given, it will compare if the hop probability
            is changing by the state itself"""
            #todo rename the meth type because now it can be chi or methyl
            #todo this function can be rebuilt anyway
            dt = self.universe.trajectory.dt
            if not hoptype=="chi1":
                hop_state_0 = np.zeros(self.n, dtype=self.default_float)
                hop_state_1 = np.zeros(self.n, dtype=self.default_float)
                hop_state_2 = np.zeros(self.n, dtype=self.default_float)
                chi_areas = self.chi1_areas if chi_type == 1 else self.chi2_areas
                hops = self.hops if "methyl" in hoptype else self.hops_chi2 if "chi" in hoptype else None
                for i in range(self.n):
                    hop_state_0[i] = (hops[index, i * part:(i + 1) * part] *
                                      (chi_areas[chi_num, i * part:(i + 1) * part] == 0)).sum() / part
                    hop_state_1[i] = (hops[index, i * part:(i + 1) * part] *
                                      (chi_areas[chi_num, i * part:(i + 1) * part] == 1)).sum() / part
                    hop_state_2[i] = (hops[index, i * part:(i + 1) * part] *
                                      (chi_areas[chi_num, i * part:(i + 1) * part] == 2)).sum() / part

                axis.bar(np.arange(n) + .5, (hop_state_0/dt + hop_state_1/dt + hop_state_2/dt), color="b")
                axis.bar(np.arange(n) + .5, (hop_state_0/dt + hop_state_1/dt), color="g")
                axis.bar(np.arange(n) + .5, (hop_state_0/dt), color="r")
                #axis.bar(np.arange(n) + .5, (hop_state_0 + hop_state_1 + hop_state_2), color="b")
                #axis.bar(np.arange(n) + .5, (hop_state_0 + hop_state_1), color="g")
                #axis.bar(np.arange(n) + .5, (hop_state_0), color="r")
                #axis.set_yscale("log")

            else:
                hops = self.hops_chi1
                hop_state = np.zeros(self.n, dtype=self.default_float)
                for i in range(self.n):
                    hop_state[i] = hops[index, i*part:(i+1)*part].sum()/part
                axis.bar(np.arange(n)+.5, hop_state/dt, color="black")
            axis.set_xlim(0, n)
            axis.set_ylabel(r'$\langle \tau_{hop}^{-1} \rangle ps^{-1}$')
            #axis.set_ylabel(r'$P\tau_{hop, 5ps}$')
            axis.set_xlabel("µs",labelpad=-3)
            if hoptype == "methyl":
                #axis.text(self.n / 5, axis.get_ylim()[1] * .8,
                #      r'$CH_3^{{{}}} with \chi_{{{}}}-states$'.format(meth_type, chi_type), bbox=TEXTBOX)
                axis.set_title(r'$CH_3^{{{}}}\ hop\ correlation\ rate\ depending\ on\ \chi_{{{}}}-states$'.format(meth_type, chi_type))
            elif hoptype =="chi2":
                #axis.text(self.n / 5, axis.get_ylim()[1] * .8,
                #          r'$\chi_2 with \chi_1-states$', bbox=TEXTBOX)
                axis.set_title(r'$\chi_2 \ hop\ correlation\ rate\ depending\ on\ \chi_1-states$')
            else:
                axis.set_title(r'$\chi_1 \ hop\ correlation\ rate$')

            n_ticks = 5
            axis.set_xticks(range(0,self.n+1,self.n//(n_ticks-1)))
            axis.set_xticklabels([f"{x:.2f}" for  x in np.linspace(0,10,n_ticks)])

            #TODO remove
            #if meth_type==2:
            #    axis.set_ylim(0,1)
            #if hoptype != "methyl":
            #    axis.set_ylim(0,0.2)
            #todo remove

        def plot_detectors():
            return
            def tc_str(z):
                unit = 's'
                if z <= -12: z, unit = z + 15, 'fs'
                if z <= -9: z, unit = z + 12, 'ps'
                if z <= -6: z, unit = z + 9, 'ns'
                if z <= -3: z, unit = z + 6, r'$\mu$s'
                if z <= 0: z, unit = z + 3, 'ms'
                tc = np.round(10 ** z, -1 if z >= 2 else (0 if z >= 1 else 1))
                return '{:2} '.format(tc if tc < 10 else int(tc)) + unit
            D = DR.data()
            cts = self.cts[dix[key]["ct_indices"]]
            D.load(Ct={'Ct':cts
                       ,'t': np.linspace(0,int(self.length/1000)*self.universe.trajectory.dt, self.length)})
            n_dets = self.n_dets if self.n_dets else 8
            remove_S2 = 1
            if not hasattr(self, "detect"):
                D.detect.r_auto3(n=n_dets)
                self.detect = D.detect
            else:
                D.detect = self.detect

            fit = D.fit()
            tick_labels = ['~' + tc_str(z0) for z0 in fit.sens.info.loc['z0']]
            tick_labels[0] = '<' + tick_labels[1][1:]
            #tick_labels[-1] = '>' + tick_labels[-2][1:]
            tick_labels = tick_labels[:n_dets-1]  # removed last label because no data were there
            page_2 = plt.figure()
            h = int((cts.shape[0]+1)/2)+1
            page_2.set_size_inches(8.3, 11.7/8*h)  # reduce page size for less cts
            ax0 = page_2.add_subplot(h, 1, 1)
            ax0.plot(D.sens.z(), fit.sens.rhoz().T)
            ax0.set_xlim(-14, -3)
            ax0.set_title("Detector Sensitivities")
            ax0.set_ylabel(r'$(1-S^2)$')
            ax0.set_xlabel(r'$\log_{10}(\tau_c/s)$')
            ax0.set_yticks([0, 1])
            ax = [page_2.add_subplot(h, 2, 3 + i) for i in range(cts.shape[0])]
            for d in range(n_dets - remove_S2):
                for r in range(cts.shape[0]):
                    ax[r].bar(d, fit.R[r, d])
            for r in range(cts.shape[0]):
                maxR = np.max(fit.R[r, :-remove_S2])
                ylim = np.round(maxR + .06,1) if maxR < .5 else .75 if maxR < .72 else 1
                ax[r].set_ylim(0, ylim)
                ax[r].text(n_dets/2, ylim * .75, dix[key]["ct_labels"][r], bbox=TEXTBOX)
                ax[r].set_xticks(range(n_dets - remove_S2))
                if r < cts.shape[0]-2:
                    ax[r].set_xticklabels([])
                else:
                    ax[r].set_xticklabels(tick_labels, rotation=66)
                if r % 2 == 0:
                    ax[r].set_ylabel(r'$\rho_n^{(\Theta,S)}$')
            page_2.suptitle("{}_{}".format(key, self.sel_sim_name))
            page_2.tight_layout()
            pdf.savefig(page_2)
            plt.close(page_2)
        dix = self.full_dict
        part = self.part
        n = self.n
        if not exists(join(self.dir, "plots")):
            mkdir(join(self.dir, "plots"))

        for _, key in enumerate(dix):
            #if not "ILE" in key and not "VAL" in key and not "LEU" in key:
            #    continue
            with PdfPages(join(self.dir, "plots", "{}_{}.pdf".format(key, self.sel_sim_name))) as pdf:
                num = dix[key][dix[key]["labels"][0]]["index"]  # index of the first (or only) methylgroup
                if len(dix[key]["labels"]) == 2:
                    two = True
                    num2 = dix[key][dix[key]["labels"][1]]["index"]  # index of the second methylgroup
                    #print(dix[key][dix[key]["labels"][1]])
                else:
                    two = False
                chi1 = dix[key].get("chi1")  # index for chi1
                chi2 = dix[key].get("chi2")  # index for chi2
                ### Initializing figure and axes
                page_1 = plt.figure()
                page_1.set_size_inches(8.3, 11.7)  # (14, 10)
                if chi1 is not None or chi2 is not None:
                    ax_3d_plot = page_1.add_subplot(321, projection='3d')  # for plotting 3d distribution of methyl hydrogens
                    rama_ax = page_1.add_subplot(322)
                    if two:  # check if two methylgroups are available in the residue
                        hopability_methyl_1_axis = page_1.add_subplot(625)  # if yes, create two hopability plots
                        hopability_methyl_2_axis = page_1.add_subplot(627)
                        if chi2 is not None:
                            hopability_methyl_1_axis_B = page_1.add_subplot(6, 2, 9)  # if yes, create two hopability plots
                            hopability_methyl_2_axis_B = page_1.add_subplot(6, 2, 11)
                    else:
                        hopability_single_methyl_axis = page_1.add_subplot(323)  # if not, only one (surprise)
                    if chi2 is not None:  # same here if chi1 and chi2 are available
                        ct_plot = page_1.add_subplot(326)
                        chi1_hopability_plot = page_1.add_subplot(626)  # then make two plots
                        chi2_hopability_plot = page_1.add_subplot(628)
                        pie_chi2 = page_1.add_subplot(645)
                        pie_chi1 = page_1.add_subplot(646)
                        pie_chi2.pie([(self.chi2_areas[chi2]==0).sum(), (self.chi2_areas[chi2]==1).sum(),
                                      (self.chi2_areas[chi2]==2).sum()], colors=["r", "g", "b"])
                        pie_chi2.text(0,1.25,r'$\chi_2-states:$', ha="center", va="center", bbox=TEXTBOX)
                        #pie_chi2.set_title(r'$\chi_2$-states:')
                    else:
                        ct_plot = page_1.add_subplot(313)
                        chi1_hopability_ax_solo = page_1.add_subplot(324)  # or only one
                        #tau_legend.append(r'$\chi_1 hops$')
                        pie_chi1 = page_1.add_subplot(645)
                    pie_chi1.pie([(self.chi1_areas[chi1]==0).sum(), (self.chi1_areas[chi1]==1).sum(),
                                  (self.chi1_areas[chi1]==2).sum()], colors=["r", "g", "b"])
                    pie_chi1.text(0,1.25,r'$\chi_1-states:$', ha="center", va="center", bbox=TEXTBOX)
                    #pie_chi1.set_title(r'$\chi_1$-states:', pad=10)
                else:
                    # this part is mostly for Alanine
                    ax_3d_plot = page_1.add_subplot(321, projection='3d')
                    ct_plot = page_1.add_subplot(313)
                    hopability_single_methyl_axis = page_1.add_subplot(312)  # if not, only one (surprise)
                ### start to plot, first the 3d scatter plot of the residue chain
                ax_3d_plot.set_axis_off()
                ax_3d_plot.scatter(self.coords_3Dplot[num, :, 0], self.coords_3Dplot[num, :, 1],
                                   self.coords_3Dplot[num, :, 2], s=.5, c="g")  # areas[num])

                carbons = np.array([get_x_y_z_old(dix[key]["C"].position, dix[key]["CA"].position,
                                                  dix[key]["CB"].position, dix[key][_].position) for _ in
                                    dix[key]["chain"]])
                ax_3d_plot.plot(*carbons.T, marker="o", markersize=10)
                for c, label in zip(carbons, dix[key]["chain"]):
                    ax_3d_plot.text(*c.T, label)
                if two:
                    carbon = dix[key][dix[key]["labels"][0]]['carbon']
                    ax_3d_plot.text(*get_x_y_z_old(dix[key]["C"].position, dix[key]["CA"].position,
                                                  dix[key]["CB"].position,carbon.position),r'$CH_3^1:$'+carbon.name,bbox=TB2)

                    carbon = dix[key][dix[key]["labels"][1]]['carbon']
                    ax_3d_plot.text(*get_x_y_z_old(dix[key]["C"].position, dix[key]["CA"].position,
                                                  dix[key]["CB"].position,carbon.position),r'$CH_3^2:$'+carbon.name,bbox=TB2)
                ax_3d_plot.view_init(elev=0, azim=45)
                if len(dix[key]["ct_indices"]):
                    ct_plot.semilogx(np.arange(1, self.length+1),
                                     self.cts[dix[key]["ct_indices"]].T)
                    for l,index in enumerate(dix[key]["ct_indices"]):
                        dix[key]["ct_labels"][l] += " S²={:.2f}".format(self.S2s[index])
                    ct_plot.legend(dix[key]["ct_labels"])
                    ct_plot.set_xlabel("timepoints")
                    # todo calculate the correlation time by dt and stuff and set the xticks
                    ct_plot.set_ylabel("C(t)")
                    ct_plot.set_yticks([0, .5, 1])
                    ct_plot.set_ylim(-.1, 1)
                    ct_plot.set_xticks([1+self.length/1000000,1+self.length/100000,1+self.length/10000,self.length/1000,self.length/100,self.length/10,self.length])
                    mmax = int((self.universe.trajectory.dt*self.length)//1000000)
                    ct_plot.set_xticklabels([f"{mmax}ps",f"{mmax*10}ps",f"{mmax*100}ps",f"{mmax}ns",f"{mmax*10}ns",f"{mmax*100}ns",f"{mmax}µs"])
                    ct_plot.set_title("Correlation functions")

                if two:  # todo this is still clumsy
                    ax_3d_plot.scatter(self.coords_3Dplot[num2, :, 0], self.coords_3Dplot[num2, :, 1],
                                       self.coords_3Dplot[num2, :, 2], s=.5, c="b")  # 3D Rama plot
                    if chi2 is not None:
                        plot_hopability_with_chi_states(hopability_methyl_1_axis, num, 1, chi2, 2)
                        plot_hopability_with_chi_states(hopability_methyl_2_axis, num2, 2, chi2, 2)
                        plot_hopability_with_chi_states(hopability_methyl_1_axis_B, num, 1, chi1, 1)
                        plot_hopability_with_chi_states(hopability_methyl_2_axis_B, num2, 2, chi1, 1)
                    elif chi1 is not None:
                        plot_hopability_with_chi_states(hopability_methyl_1_axis, num, 1, chi1, 1)
                        plot_hopability_with_chi_states(hopability_methyl_2_axis, num2, 2, chi1, 1)
                    else:
                        hopability_methyl_1_axis.bar(np.arange(n), self.hopability[num, :], color="black")
                        hopability_methyl_2_axis.bar(np.arange(n), self.hopability[num2, :], color="black")
                else:
                    hopability_single_methyl_axis.bar(np.arange(n) + .5, self.hopability[num], color="black")  # hopability
                    hopability_single_methyl_axis.set_xlim(0, n)
                    if chi1 is not None:
                        plot_hopability_with_chi_states(hopability_single_methyl_axis, num, 1, chi1, 1)

                if chi1 is not None or chi2 is not None:
                    if chi2 is not None:
                        rama_ax.hist2d((self.dih_chi1[chi1] + 240) % 360, (self.dih_chi2[chi2] + 240) % 360,
                                       range=[[0, 360], [0, 360]], bins=[180, 180], cmin=0.1)
                        rama_ax.set_xlabel(r'$\chi_1$')
                        rama_ax.set_ylabel(r'$\chi_2$')
                        rama_ax.set_title("Ramachandran")
                        rama_ax.set_ylim(0, 360)
                        rama_ax.set_yticks([0, 60, 180, 300, 360])
                        plot_hopability_with_chi_states(chi1_hopability_plot, chi1, 2, chi1, 1, "chi1")
                        #chi1_hopability_plot.bar(np.arange(n) + .5, self.hopability_chi1[chi1], color="black")
                        #chi1_hopability_plot.set_xlim(0, n)
                        #chi1_hopability_plot.set_xticklabels([])
                        #chi1_hopability_plot.text(n / 10, chi1_hopability_plot.get_ylim()[1] * .8,
                        #                          r'$\chi_1 hop-probability$', bbox=TEXTBOX)
                        plot_hopability_with_chi_states(chi2_hopability_plot,chi2,2,chi1,1,"chi2")
                    else:
                        #chi1_hopability_ax_solo.bar(np.arange(n) + .5, self.hopability_chi1[chi1], color="black")
                        #chi1_hopability_ax_solo.set_xlim(0, n)
                        #chi1_hopability_ax_solo.text(n / 10, chi1_hopability_ax_solo.get_ylim()[1] * .8,
                        #                             r'$\chi_1 hop-probability$', bbox=TEXTBOX)
                        plot_hopability_with_chi_states(chi1_hopability_ax_solo, chi1, 2, chi1, 1, "chi1")
                        rama_ax.hist((self.dih_chi1[chi1] + 240) % 360, bins=range(0, 120, 3),
                                     range=range(120), color="r")
                        rama_ax.hist((self.dih_chi1[chi1] + 240) % 360, bins=range(120, 240, 3),
                                     range=range(120), color="b")
                        rama_ax.hist((self.dih_chi1[chi1] + 240) % 360, bins=range(240, 360, 3),
                                     range=range(120), color="g")
                        rama_ax.set_xlabel(r'$\chi_1$')
                        rama_ax.set_ylabel("occurrence")
                        rama_ax.set_title(r'$\chi_1$ \ angle \ histogram')
                        rama_ax.set_yticklabels([])

                    # final configuration of the plots
                    rama_ax.set_xlim(0, 360)
                    rama_ax.set_xticks([0, 60, 180, 300, 360])


                ct_plot.set_xlim(1, self.length)
                page_1.suptitle("{} of simulation {} with\n {} µs and dt of {}ps".format(key, self.sel_sim_name,
                                    (self.length*self.universe.trajectory.dt)//1000000, int(self.universe.trajectory.dt)))
                page_1.tight_layout()
                page_1.subplots_adjust(left=.065,right=.99, wspace=0.2,hspace=0.33)
                pdf.savefig(page_1)
                plt.close(page_1)
                plot_detectors()
                print("Save", key, "plot")


#@time_runtime
def main():
    M = KaiMarkov(simulation=4, n=100)
    M.calc(sparse=0)
    M.plot_all()

def alt():
    for i in [6]:#range(6):
        if i == 0:
            M = KaiMarkov(simulation=i,offset=0)
        else:
            M = KaiMarkov(simulation=i)

        with open(f"{M.sel_sim_name}_chi_probabilities","w+") as f:
            M.calc(sparse=0)
            #M.plot_all()
            for res in M.full_dict.keys():
                try:
                    chi1_idx = M.full_dict[res]["chi1"]
                except:
                    print("Residue",res, "does not have chi1")
                    chi1_idx = None

                try:
                    chi2_idx = M.full_dict[res]["chi2"]
                except:
                    print("Residue",res, "does not have chi2")
                    chi2_idx = None
                if chi1_idx is None and chi2_idx is None:
                    continue
                f.write(f"{res};")
                print(res, chi1_idx, chi2_idx)
                if chi1_idx is not None:

                    chi1 = M.dih_chi1[chi1_idx]
                    chi1_areas = np.zeros(chi1.shape)
                    chi1_areas += ((chi1 >= 0) == (chi1 < 120)).astype('uint8')
                    chi1_areas += ((chi1 < 0) == (chi1 >= -120)).astype('uint8') * 2
                    f.write("Chi1[{:.3f},{:.3f},{:.3f}]".format((chi1_areas==0).sum()/chi1.shape[0],(chi1_areas==1).sum()/chi1.shape[0],(chi1_areas==2).sum()/chi1.shape[0]))
                if chi2_idx is not None:
                    chi2 = M.dih_chi2[chi2_idx]
                    chi2_areas = np.zeros(chi2.shape)
                    chi2_areas += ((chi2 >= 0) == (chi2 < 120)).astype('uint8')
                    chi2_areas += ((chi2 < 0) == (chi2 >= -120)).astype('uint8') * 2
                    f.write(";Chi2[{:.3f},{:.3f},{:.3f}]".format((chi2_areas == 0).sum() / chi2.shape[0], (chi2_areas == 1).sum() / chi2.shape[0],
                          (chi2_areas == 2).sum() / chi2.shape[0]))
                f.write("\n")


    return

    res = "ILE_256"
    chi1 = M.full_dict[res]["chi1"]
    chi2 = M.full_dict[res]["chi2"]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist2d((M.dih_chi1[chi1]+ 240) % 360,(M.dih_chi2[chi2]+ 240) % 360, bins = (360,360),cmin=1)
    ax.set_xlim(0,360)
    ax.set_ylim(0,360)
    ax.set_xlabel('$\chi_1[°]$')
    ax.set_ylabel('$\chi_2[°]$')
    ax.set_xticks([60,180,300])
    ax.set_yticks([60,180,300])
    fig.tight_layout()
    plt.show()

    return


    for i in range(5):
        M = KaiMarkov(simulation=i, n=100)
        M.calc(sparse=1)
        M.plot_all()
    return
    '''
    M = KaiMarkov(n=200,simulation=4,residues=[])
    M.calc(sparse=0)
    M.plot_all()
    M = KaiMarkov(n=200, simulation=2, residues=[])
    M.calc(sparse=0)
    M.plot_all()

    return
    M = KaiMarkov(n=500,  simulation=2, residues=[])
    M.calc(sparse=10)
    M.universe.trajectory[0]
    #M.plot_all()
    #return
    '''
    M = KaiMarkov(n=400)
    M.calc()
    M.universe.trajectory[0]
    fig = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    ax = fig.add_subplot(111,projection="3d")
    bx = fig2.add_subplot(111,projection="3d")
    cx = fig3.add_subplot(111, projection="3d")
    with open("MetRotCorr.bild","w+") as f:
        for key in M.full_dict.keys():
            for label in M.full_dict[key]["labels"]:
                pos = M.full_dict[key][label]["carbon"].position
                ax.scatter(*pos,color="black")
                ax.text(*pos,label)
                num = M.full_dict[key][label]["index"]
                for key2 in M.full_dict.keys():
                    for label2 in M.full_dict[key2]["labels"]:
                        if label ==  label2:
                            continue
                        pos2 = M.full_dict[key2][label2]["carbon"].position
                        num2 = M.full_dict[key2][label2]["index"]
                        corr = np.corrcoef(M.hopability[num],M.hopability[num2])[1][0]
                        if np.linalg.norm(pos-pos2) < 20 and np.abs(corr)>.35:
                            ax.plot([pos[0],pos2[0]],[pos[1],pos2[1]],[pos[2],pos2[2]], color =((np.abs(corr),0,0) if corr < 0 else (0,np.abs(corr),0)),
                                    alpha=corr**2)
                            f.write(".color {} {} {}\n".format(np.abs(corr) if corr < 0 else 0, corr if corr > 0 else 0, 0))
                            #f.write(".transparency {}\n".format(np.abs(corr)))
                            f.write(".cylinder {} {} {} {} {} {} {}\n".format(pos[0],pos[1],pos[2],pos2[0],pos2[1],pos2[2],np.abs(corr-.2)/4))
                chi1 = M.full_dict[key].get("chi1")
                if chi1 is not None:
                    corr = np.corrcoef(M.hopability[num],M.hopability_chi1[chi1])[1][0]
                    bx.scatter(*pos, color=(np.abs(corr),0,0) if corr < 0 else (0,corr,0), s=np.abs(corr)*100)
                    bx.text(*pos, label)
                chi2 = M.full_dict[key].get("chi2")
                if chi2 is not None:
                    corr = np.corrcoef(M.hopability[num],M.hopability_chi2[chi2])[1][0]
                    cx.scatter(*pos, color=(np.abs(corr),0,0) if corr < 0 else (0,corr,0), s=np.abs(corr)*100)
                    cx.text(*pos, label)



if __name__ == "__main__":
    M = KaiMarkov(simulation=0,n=200)
    M.calc()
    M.plot_all()

