import glob

train = 'data_models/labels/train/*'
val = 'data_models/labels/val/*'

hat, glass = 0, 0

for i in glob.glob(train) + glob.glob(val):
    with open(i) as file:
        lines = file.readlines()
        for line in lines:
            label = line.split(' ')[0]
            if int(label) == 0:
                hat += 1
            else:
                glass += 1
            print(line)
            print(label)
            
    
    

print(f'hat total: {hat}')
print(f'glass total: {glass}')