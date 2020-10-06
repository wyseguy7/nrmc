import copy
import os
import subprocess
import sys

runhuh = ("--run" in sys.argv)

replace = "main"
template = replace + ".sh"

processes = ["single_node_flip_tempered", "district_to_district", 
           "center_of_mass", "single_node_flip"]
steps = 10000000
runsPerProcess = 3

# processes = ["single_node_flip_tempered", "district_to_district", 
#             "center_of_mass", "single_node_flip"]
# steps = 1000
# runsPerProcess = 1

replace = "jobname"
templateLines = open(template).readlines()
if '--mecklenburg' in sys.argv:
    runsPerProcess = 10
    district_list = [5]
    n_list = [0]
    folder = "--folder data/Mecklenburg/"
    process_name = "mecklenburg"
    scoring = "--score_func compactness population_balance --score_weights 0.5 0.00000001"
    apd = 1.0

else:
    district_list = [3,4]
    runsPerProcess = 3
    n_list = [10, 20, 40]
    folder = ""
    process_name = "lattice"
    scoring = "--score_func cut_length --score_weights 1.0"
    apd = 0.1


for process in processes:
    for num_districts in district_list:
        for n in n_list:
            for runInd in range(runsPerProcess):
                jobname = process + "_runIndex_" + str(runInd) + "_steps_" + str(steps) + "_num_districts_{}_n_{}_{}".format(num_districts,n, process_name)
                foutStr = "." + jobname + ".sh"
                fout = open(foutStr, "w")
                for ii in range(len(templateLines)):
                    line = templateLines[ii]
                    outPath = os.path.join(os.sep, "gtmp", "etw16", "runs", process_name, process,
                                           "num_districts={}n={}".format(num_districts,n), "runInd_" + str(runInd))
                    # outPath = os.path.join(".", "runs", process,
                    #                        "runInd_" + str(runInd))
                    if not os.path.exists(outPath):
                        os.makedirs(outPath)
                    if "python src/scripts/run.py" in line:
                        line = line.rstrip()
                        line += ' ' + ' '.join(["--steps", str(steps)]) + ' '
                        line += ' ' + ' '.join(["--process", process]) + ' '
                        line += ' ' + "--output_path " + outPath + " " +\
                                "--num_districts {nd} {scoring} --apd {apd} --n {n} {folder}\n".format(nd=num_districts, n=n, folder=folder, scoring=scoring, apd=apd)

                    line = line.replace("jobname", jobname)
                    fout.write(line)
                fout.close()
                if runhuh:
                    print("running", jobname)
                    subprocess.Popen(['qsub', foutStr])

