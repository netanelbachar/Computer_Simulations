#%% imports
## THIS CODE ENTERS IN YOUR COMPUTER TO THE TERMINAL AND CHANGES INPUTFILES FROM ONE LINE TO ANOTHER
import numpy as np
import subprocess
#%% some variables - may change from run to run
bhw1 = np.array([3, 4, 6, 10])
beads1 = np.array([4,8,12,16,32,64])

bhw2 = np.array([30,60])
beads2 = np.array([1,8,16,32,64,80,96])

old_loop = 3
new_loop = 10

old_x = 33.0 # for 3 bosons
new_x = 10.0 # for 10 bosons

old_lambda = 9.53158e-11 # lambda_kfir
new_lambda = 1.00e-10 # arbitrary for now

old_name = 'kfir'
new_name = 'big' # or small (depends on lambda)

#%% CODE TEMPLATE
# for bhw or beta 30,60 need to change bhw1 to bhw2 and beads1 to beads2 every time they appear in the code
'''
for i in range(len(bhw1)):
    for j in range(len(beads1)):    
        subprocess.run(['command','add_ons to command']) # every word in the command should be in a different list index (qstat -u will be written as ['qsub','-u'])
'''
#%% CODE EXAMPLES
# to change from M bosons to N bosons - both input and runfile need changes for both bhw1 and bhw2 - HARMONIC PATH 

# sed -i is a command that changes a line in the code from a to b, with delimiter of / or @ or _ usually (sed -i 's/a/b/' 'file_path')

# the format is the command that takes the correct variables for the loop (instead of running 10 commands, 1 for each bead number, it runs it in a loop over all beads in the list beads1 or beads2)


# for bhw1
for i in range(len(bhw1)):
    for j in range(len(beads1)):
        # changing the input file
        subprocess.run(['sed', '-i', 's/replicate 2 2 1/replicate 1 2 1/', '/home/netanelb2/Desktop/Netanel/Research/copy_paste/boson10/bhw{fbhw}/{fbeads}beads/input.PIMD_exiton'.format(fbhw=bhw1[i], fbeads=beads1[j])])
        # changing the runfile
        subprocess.run(['sed', '-i', 's/variable a loop 3/variable a loop 10/', '/home/netanelb2/Desktop/Netanel/Research/copy_paste/boson10/bhw{fbhw}/{fbeads}beads/input.PIMD_exiton'.format(fbhw=bhw1[i], fbeads=beads1[j])])
        print ("bhw=", bhw1[i], "fbeads=", beads1[j])
        # subprocess.run(['sed','-i','s@RUNDIR=/hirshblab-storage/netanelb2/PIMD/exitons/moire_four/boson{fold}/bhw{fbhw}/{fbeads}beads@RUNDIR=/hirshblab-storage/kfirvolpin/research_project/harmonic_pimd/bosons/boson{fnew}/bhw{fbhw}/{fbeads}beads/@'.format(fold=old_loop, fnew=new_loop, fbhw=bhw2[i], fbeads=beads2[j]),'/home/kfir/research_project/harmonic/template/bhw{fbhw}/{fbeads}beads/run_lmp_local.sh'.format(fbhw = bhw1[i],fbeads = beads1[j])])


# for bhw2
for i in range(len(bhw2)):
    for j in range(len(beads2)):
        # changing input file
        subprocess.run(['sed', '-i', 's/replicate 2 2 1/replicate 1 2 1/', '/home/netanelb2/Desktop/Netanel/Research/copy_paste/boson10/bhw{fbhw}/{fbeads}beads/input.PIMD_exiton'.format(fbhw=bhw2[i], fbeads=beads2[j])])
#         # changing runfile
        subprocess.run(['sed', '-i', 's/variable a loop 3/variable a loop 10/', '/home/netanelb2/Desktop/Netanel/Research/copy_paste/boson10/bhw{fbhw}/{fbeads}beads/input.PIMD_exiton'.format(fbhw=bhw2[i], fbeads=beads2[j])])
#         subprocess.run(['sed','-i','s@RUNDIR=/hirshblab-storage/kfirvolpin/research_project/harmonic_pimd/bosons/boson{fold}/bhw{fbhw}/{fbeads}beads@RUNDIR=/hirshblab-storage/kfirvolpin/research_project/harmonic_pimd/bosons/boson{fnew}/bhw{fbhw}/{fbeads}beads/@'.format(fold = old_loop,fnew = new_loop,fbhw = bhw2[i],fbeads = beads2[j]),'/home/kfir/research_project/harmonic/template/bhw{fbhw}/{fbeads}beads/run_lmp_local.sh'.format(fbhw = bhw2[i],fbeads = beads2[j])])


# to change lambda - both input and runfile need changes
'''
Nbosons = 10
for i in range(len(bhw1)):
    for j in range(len(beads1)):
        # changing input file
        subprocess.run(['sed','-i','s/variable k equal {fold}/variable k equal {fnew}/'.format(fold = old_lambda,fnew = new_lambda),'/home/kfir/research_project/anharmonic/template/beta{fbhw}/{fbeads}beads/input.PIMD_exiton'.format(fbhw = bhw1[i],fbeads = beads1[j])])
        # changing runfile
        subprocess.run(['sed','-i','s@RUNDIR=/hirshblab-storage/kfirvolpin/research_project/anharmonic/lambda_{fold}/boson{fboson}/beta{fbhw}/{fbeads}beads@RUNDIR=/hirshblab-storage/kfirvolpin/research_project/anharmonic/lambda_{fnew}/boson{fboson}/beta{fbhw}/{fbeads}beads/@'.format(fold = old_name,fnew = new_name,fboson = Nbosons,fbhw = bhw1[i],fbeads = beads1[j]),'/home/kfir/research_project/anharmonic/template/beta{fbhw}/{fbeads}beads/run_lmp_local.sh'.format(fbhw = bhw1[i],fbeads = beads1[j])])

for i in range(len(bhw2)):
    for j in range(len(beads2)):
        # changing input file
        subprocess.run(['sed','-i','s/variable k equal {fold}/variable k equal {fnew}/'.format(fold = old_lambda,fnew = new_lambda),'/home/kfir/research_project/anharmonic/template/beta{fbhw}/{fbeads}beads/input.PIMD_exiton'.format(fbhw = bhw2[i],fbeads = beads2[j])])
        # changing runfile
        subprocess.run(['sed','-i','s@RUNDIR=/hirshblab-storage/kfirvolpin/research_project/anharmonic/lambda_{fold}/boson{fboson}/beta{fbhw}/{fbeads}beads@RUNDIR=/hirshblab-storage/kfirvolpin/research_project/anharmonic/lambda_{fnew}/boson{fboson}/beta{fbhw}/{fbeads}beads/@'.format(fold = old_name,fnew = new_name,fboson = Nbosons,fbhw = bhw2[i],fbeads = beads2[j]),'/home/kfir/research_project/anharmonic/template/beta{fbhw}/{fbeads}beads/run_lmp_local.sh'.format(fbhw = bhw2[i],fbeads = beads2[j])])
'''
