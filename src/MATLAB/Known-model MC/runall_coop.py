import sys, os, re, subprocess, time

results = sys.argv[1]
seeds = sys.argv[2]
horizon = sys.argv[3]
k_factor = sys.argv[4]
mct = sys.argv[5]
num_mct = sys.argv[6]



if not os.path.exists(results):
    os.makedirs(results)
    print(f"Created results directory {results}")

if not os.path.exists(f"{results}/logs"):
    os.makedirs(f"{results}/logs")
    print(f"Created results-logs directory {results}/logs")


ssub_path = '/media/labs/rsmith/lab-members/nhakimi/known_model_v2_2024/run_coop.ssub'

for subject in range(int(seeds)):
    stdout_name = f"{results}/logs/{subject}-%J.stdout"
    stderr_name = f"{results}/logs/{subject}-%J.stderr"
    jobname = 'hill_ai_navid'
    os.environ['seed'] = str(subject)
    os.environ['horizon'] = str(horizon)
    os.environ['k_factor'] = str(k_factor)
    os.environ['mct'] = str(mct)
    os.environ['num_mct'] = str(num_mct)


    os.system(f"sbatch -J {jobname} -o {stdout_name} -e {stderr_name} {ssub_path} {results}")
    print(f"SUBMITTED JOB [{jobname}]")


    ###python3 runall_coop.py "/media/labs/rsmith/lab-members/nhakimi/known_model_v2_2024/results" 100 1 0.7 6 100
    # srun -N 1 -c 4 --mem=50000 --pty --x11 --partition=c2_short /bin/bash
    # cd /media/labs/rsmith/lab-members/nhakimi/known_model_v2_2024/
    #    joblist | grep hill_ai_navid | grep -Po 131.... | xargs scancel
