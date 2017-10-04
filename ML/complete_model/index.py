from train_test_generator import train_test_generator
from set_divide_model_generator import set_divide_model_generator
from sets_model_generator import sets_model_generator
from main_model import main_model
from setlist import setlist
import sys
def update_progress(progress,n):
    hashes = '#'*int(progress*40/n)
    blanks = ' '*(40-len(hashes))
    percent = (progress*100)/n
    sys.stdout.write('\r[{0}] {1}%'.format(hashes+blanks, percent))

n = 10
generator = train_test_generator()
set_divide_model = set_divide_model_generator()
sets_model = sets_model_generator()
main = main_model()

set_divide_acc = 0
sets_model_acc = [0 for x in range(len(setlist)) if len(setlist[x]) > 1]
final_acc = 0
print("\n")
update_progress(0,n)
for test in range(n):
    generator.generate()

    set_divide_acc += set_divide_model.train()

    acc = sets_model.train()
    for i in range(len(acc)):
        sets_model_acc[i] += acc[i]

    final_acc += main.train()
    update_progress(test+1,n)
set_divide_acc = set_divide_acc/n
for i in range(len(sets_model_acc)):
    sets_model_acc[i] = sets_model_acc[i]/n
final_acc = final_acc/n
print("\n")
print("Set Divide Accuracy: "+str(set_divide_acc))
print("\n")
for i in range(len(sets_model_acc)):
    print("Set "+str(i)+" Accuracy: "+str(sets_model_acc[i]))
print("\n")
print("Final Accuracy: "+str(final_acc))
