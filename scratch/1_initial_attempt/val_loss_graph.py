import pandas as pd
import matplotlib.pyplot as plt

history = pd.read_csv('training_history.csv', skiprows=3)
print(history.iloc[:, 0])
print(history.iloc[:, 3])

plt.plot( history.iloc[:, 3])
plt.xlabel('Epoch')
plt.ylabel('Val loss')
plt.show()
plt.savefig('training_loss.png')