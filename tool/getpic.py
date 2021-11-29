import matplotlib.pyplot as plt
import numpy as np


def getPic(xTick, cleanAcu, fgsmAcu, pgdAcu, deepfoolAcu, cwAcu, x, y, model, loss, attack):
    plt.figure()
    plt.title('NatAcu and RobAcu of AT')
    plt.plot(xTick, cleanAcu, color='green', label='Clean_Accuracy')
    plt.plot(xTick, fgsmAcu, color='red', label='FGSM_Accuracy')
    plt.plot(xTick, pgdAcu, color='yellow', label='PGD_Accuracy')
    plt.plot(xTick, deepfoolAcu, color='blue', label='DeepFool_Accuracy')
    plt.plot(xTick, cwAcu, color='purple', label='C&W_Accuracy')
    plt.xticks(np.arange(0, x, 10))
    plt.yticks(np.arange(0, y, .1))
    plt.xlim(0, 70)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.savefig("/tmp/Adversarial_Example/images/img-"+model+"-"+loss+"-"+attack+".png")
    plt.show()

