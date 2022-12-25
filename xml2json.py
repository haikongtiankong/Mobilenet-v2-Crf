import sys
import os
import argparse
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../')
from Annotation import Formatter  # noqa

'''
Transform xml file to json file that includes coordinates of tumor areas
'''
parser = argparse.ArgumentParser(description='Convert Camelyon16 xml format to'
                                 'internal json format')
parser.add_argument('--xml_path', default=r"D:\self_study\medical_imaging\xml_example", metavar='XML_PATH', type=str,
                    help='Path to the input Camelyon16 xml annotation')


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    for xml in os.listdir(args.xml_path):
        xml_wholepath = os.path.join(args.xml_path,xml)
        (xml_path, xml_name_ext) = os.path.split(xml_wholepath)
        (xml_name, extension) = os.path.splitext(xml_name_ext)
        # if xml_name[-2:] == '异常':
        #     xml_name = xml_name[:-2] + 'tumor'
        # else:
        #     xml_name = xml_name[:-2] + 'normal'
        out_json = r'D:\self_study\medical_imaging\json_example\%s.json' % xml_name
        Formatter.camelyon16xml2json(xml_wholepath, out_json)


if __name__ == '__main__':
    main()
