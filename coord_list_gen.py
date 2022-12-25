import os
import re
'''
Rename the patch extracted by histolab, and also generate a txt file that includes the coordinates of each patch
'''

def patch_name_change(patch_path):
    '''修改patch命名，使遍历时可以按文件夹的顺序遍历'''
    path_list = os.listdir(patch_path)
    for patch in path_list:
        patch_id = patch.split('_')[1]
        if len(patch_id) == 1:
            patch_id = '00' + patch_id
        elif len(patch_id) == 2:
            patch_id = '0' + patch_id
        else:
            patch_id = patch_id
        patch_new = patch.split('_')[0] + '_' + patch_id + '_' + patch.split('_')[2] + '_' + patch.split('_')[3]
        os.rename(os.path.join(patch_path, patch), os.path.join(patch_path, patch_new))


def text_create(name, msg):
    desktop_path = r"D:\self_study\medical_imaging\data\003548-1b - 1\ "
    full_path = desktop_path + name + '.txt'  # 也可以创建一个.doc的word文档
    file = open(full_path, 'w')
    file.write(msg)


def get_msg_rename(slide_name,patch_path,pattern,new_name):
    msg = ''
    #new_name=250
    for patch in os.listdir(patch_path):
        patch_wholepath = os.path.join(patch_path, patch)
        (patch_path, patch_extname) = os.path.split(patch_wholepath)
        (patch_name, extension) = os.path.splitext(patch_extname)
        sign = []
        index = 0
        for idx in patch_name:
            if idx == '_':
                sign.append(index)
            if idx == '-':
                sign.append(index)
            index = index + 1

        x_ul = patch_name[sign[2] + 1:sign[3]]
        y_ul = patch_name[sign[3] + 1:sign[4]]
        x_br = patch_name[sign[4] + 1:sign[5]]
        y_br = patch_name[sign[5] + 1:]
        x_center = (int(x_ul) + int(x_br)) / 2
        y_center = (int(y_ul) + int(y_br)) / 2
        msg = msg + slide_name + ',' + str(int(x_center)) + ',' + str(int(y_center)) + '\n'

        #str_name=slide_name+'-'+str(int(x_center))+'-'+str(int(y_center))
        #if os.path.isfile(os.path.join(patch_path, patch)) == True:
        #newName = re.sub(pattern, str(idx)+'.png', patch)
        #print(newName)
        newFilename = '%d'%new_name+'.png'#patch.replace(patch, newName)
        print(newFilename)
        os.rename(os.path.join(patch_path, patch), os.path.join(patch_path, newFilename))
        new_name = new_name+1
    print('txt文本内容已生成，patch命名已修改')
    return msg, new_name


def loop(name,slide_name,patch_path,new_name):
    pattern = re.compile(r'.*')
    patch_name_change(patch_path)
    msg, new_name = get_msg_rename(slide_name, patch_path, pattern,new_name)
    text_create(name, msg)
    return new_name

def main():
    root_path = r'D:\self_study\medical_imaging\data'
    new_name = 0
    for patch in os.listdir(root_path):
        patch_path = os.path.join(root_path, patch)
        (pre_path, patch_extname) = os.path.split(patch_path)
        (slide_name, extension) = os.path.splitext(patch_extname)
        name = slide_name
        new_add = loop(name, slide_name, patch_path, new_name)
        new_name = new_add


if __name__ == '__main__':
    main()