import argparse
import os

print(os.getcwd())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cub', help='name of the dataset')
    parser.add_argument('--textPath', type=str, default=os.path.join(os.getcwd(),"cub_icml_txt"), help='text folder path')
    parser.add_argument('--imagePath', type=str, default=os.path.join(os.getcwd(),'CUB_200_2011/CUB_200_2011/images'), help='images path')
    parser.add_argument('--textNum', type=int, default=5, help='present text number')

    arguments = parser.parse_args()
    dataset, textPath, imagePath, textNum = arguments.dataset, arguments.textPath, arguments.imagePath, arguments.textNum

    html = '<html><body><h1>{} dataset</h1><table border="1px solid gray" style="width=100%"><tr><td><b>File</b></td><td><b>Caption</b></td><td><b>Image</b></td></tr>'.format(dataset)

    for sample in os.listdir(textPath):
        textClass = os.path.join(textPath, sample)
        imageClass = os.path.join(imagePath, sample)

        if not os.path.isdir(textClass):
            continue
        for textSample in os.listdir(textClass):
            textFile = os.path.join(textClass, textSample)
            imageFile = os.path.join(imageClass, textSample.replace('.txt', '.jpg'))

            folder = open(textFile, 'r')
            textList = [rawText.strip() for rawText in folder.read().split(".")]
            text = "<br>".join(textList[:textNum])

            folder.close()

            html += '\n<tr><td>{}</td><td>{}</td><td><img src="{}" width="120" height="120"/></td></tr>'.format(textSample[:-4], text, imageFile)
    
    html += '</table></body></html>'
    #html += '</html>'

    result = '{}.html'.format(dataset)
    htmlFile = open(result, 'w')
    htmlFile.write(html)
    htmlFile.close()