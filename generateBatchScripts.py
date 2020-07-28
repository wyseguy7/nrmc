import copy
import os
import subprocess
import sys

runhuh = ("--run" in sys.argv)

replace = "main"
template = replace + ".sh"

processes = ["single_node_flip_tempered", "district_to_district", 
           "center_of_mass", "single_node_flip"]
steps = 30000000
runsPerProcess = 10

# processes = ["single_node_flip_tempered", "district_to_district", 
#             "center_of_mass", "single_node_flip"]
# steps = 1000
# runsPerProcess = 1

replace = "jobname"
templateLines = open(template).readlines()

for process in processes:
    for runInd in range(runsPerProcess): 
        jobname = process + "_runIndex_" + str(runInd) + "_steps_" + str(steps)
        foutStr = "." + jobname + ".sh"
        fout = open(foutStr, "w")
        for ii in range(len(templateLines)):
            line = templateLines[ii]
            outPath = os.path.join(os.sep, "gtmp", "etw16", "runs", process, 
                                   "runInd_" + str(runInd))
            # outPath = os.path.join(".", "runs", process, 
            #                        "runInd_" + str(runInd))
            if not os.path.exists(outPath):
                os.makedirs(outPath)
            if "python src/scripts/run.py" in line:
                line = line.rstrip()
                line += ' ' + ' '.join(["--steps", str(steps)]) + ' '
                line += ' ' + ' '.join(["--process", process]) + ' '
                line += ' ' + "--output_path " + outPath + " " +\
                        "--score_func cut_length --score_weights 1.0  --n 40\n"

            line = line.replace("jobname", jobname)
            fout.write(line)
        fout.close()
        if runhuh:
            print("running", jobname)
            subprocess.Popen(['qsub', foutStr])

