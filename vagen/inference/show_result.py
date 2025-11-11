import os
import argparse

parser = argparse.ArgumentParser(description="Show inference results")
parser.add_argument("--result_dir", type=str, default="./temp_result", help="Directory containing result files")
args = parser.parse_args()
path = args.result_dir
level_count = [0,0,0,0]
level_success = [0,0,0,0]
step_success = [0,0,0,0,0,0,0,0,0,0,0]
for files in os.listdir(path):
    if not files.endswith(".txt"):
        continue
    with open(os.path.join(path,files),"r") as f:
        for line in f:
            if line.startswith("step_success"):
                _,*steps = line.strip().split(" ")
                for i,s in enumerate(steps):
                    step_success[i] += int(s)
                continue
            lvl,success,count = line.strip().split(" ")
            lvl = int(lvl)
            success = int(success)
            count = int(count)
            level_count[lvl] += count
            level_success[lvl] += success
for lvl in range(4):
    if level_count[lvl] > 0:
        print(f"  Level {lvl}: {level_success[lvl]}/{level_count[lvl]} success rate: {level_success[lvl]/level_count[lvl]:.4f}")
    else:
        print(f"  Level {lvl}: No instances")
print(f"Total: {sum(level_success)}/{sum(level_count)} success rate: {sum(level_success)/sum(level_count):.4f}")

print(f"Step-wise success distribution (for successful runs): {' '.join([str(x) for x in step_success])}")
# import matplotlib.pyplot as plt
# plt.bar(list(range(11)),step_success,label='Step-wise Success Distribution')
# plt.xlabel("Steps")
# plt.savefig("step_success_dist.png")
# for i in range(1,11):
#     step_success[i] += step_success[i-1]
# step_success = [x/sum(level_count) for x in step_success]
# plt.plot(list(range(11)),step_success,marker='o',label='Cumulative Success Rate')
# plt.xlabel("Steps")
# plt.savefig("step_success_cum.png")