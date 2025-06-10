from yaml import load, Loader

logname = 'log_v2'

header = f"""#!/usr/bin/bash
#SBATCH -G 1
#SBATCH -A m1266_g
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -C gpu 
#SBATCH -q regular 

source ./env/bin/activate
if ! [ -f "./{logname}.csv" ]; then
    echo 'tensor,ranks,init,lra,qrd,ttmc_u,lra_u,status' > {logname}.csv
fi
"""

cmd_prefix = 'python3 tucker_mixed.py'
cmds = []

with open('./experiments.yaml', 'r') as infile, open('./run.sh', 'w') as out:

    yaml_text = infile.read()
    d = load(yaml_text, Loader=Loader)

    tensors = d['tensors']

    out.write(header)

    for tensor in tensors:
        for ranks in d['tensors'][tensor]:
            rankstr = ' '.join([str(r) for r in ranks])
            for param in d['params'].values():
                record = f"{tensor},{'x'.join([str(r) for r in ranks])},{param['init']},{param['lra']},{param['qr']},{param['ttmc_u']},{param['lra_u']}"
                out.write(f"if ! grep -q \"{record},0\" \"{logname}.csv\"; then\n")
                out.write(f"\t{cmd_prefix} --tensorpath {tensor} --maxiters 10 --ntrials 10 --init {param['init']} --lra {param['lra']} --qrd {param['qr']} --ttmc_u {param['ttmc_u']} --lra_u {param['lra_u']} --tol 1e-16 --ranks {rankstr} --mode cuda\n")
                out.write(f"\techo '{record},'$? >> {logname}.csv\n")
                out.write("fi\n")


