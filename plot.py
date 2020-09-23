import numpy as np
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
avg_loss = []
with open("tiny-1c.log", "rt") as f:
    for idx, line in enumerate(f, 1):
        if "avg" in line:
            loss = line.split(" ")[2]
            avg_loss.append(loss)
batch = np.arange(1, 5001, dtype="float32")
fig = plt.figure()
plt.plot(batch, np.array(avg_loss, dtype = "float32"), 'r.-')
plt.xlabel('batch_num')
plt.ylabel('AVG_Loss')
fig.savefig('training_loss_plot.png')
plt.show()
