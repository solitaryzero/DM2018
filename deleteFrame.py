import os

for file in os.listdir("processedData/london"):
    if(file[-5] == "o"):
        continue
    temp_dir = "processedData/london/" + file
    f = open(temp_dir)
    temp = f.readlines()
    f.close()
    f = open(temp_dir,"w")
    write_str = ""
    # for i in range(len(temp)):
    #     if(i % 2 == 0):
    #         write_str += temp[i]
    # f.write(write_str)
    f.write("".join(temp[:-600]))
    print file, temp[-600]