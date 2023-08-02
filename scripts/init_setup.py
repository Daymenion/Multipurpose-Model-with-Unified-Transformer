# initialize project
import sys

# add the current directory to the imports
sys.path.insert(0, '.')
import os
# if script was executed from scripts dir then go up to root
if os.path.basename(os.path.abspath('.')) == 'scripts':
    print('[warning] init_setup: switching to repository root')
    os.chdir('..')


def print_gen(level, msg):
    print(f'[{level}] init_setup:', msg)


def print_error(msg):
    print_gen('error', msg)


def print_info(msg):
    print_gen('info', msg)


def do_task(task):
    return lambda: os.system(task)


# contains a map of directories inside data to the download sites
data_downloadables = {
    "coco": {
        "link": ["http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                 "http://images.cocodataset.org/zips/train2017.zip",
                 "http://images.cocodataset.org/zips/val2017.zip",],
        "post_download": [
            do_task("tar -xf annotations_trainval2017.zip"),
            do_task("del annotations_trainval2017.zip"),
            do_task("move annotations/* ./"),
            do_task("rmdir annotations"),
            do_task("tar -xf train2017.zip"),
            do_task("del train2017.zip"),
            do_task("move train2017 train_val"),
            do_task("tar -xf val2017.zip"),
            do_task("del val2017.zip"),
            do_task("move val2017/* train_val/"),
            do_task("rmdir val2017")
        ]
    },
    "vqa": {
        "link": [
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
        ],
        "post_download": [
            do_task("tar -xf v2_Annotations_Train_mscoco.zip"),
            do_task("del v2_Annotations_Train_mscoco.zip"),
            do_task("tar -xf v2_Annotations_Val_mscoco.zip"),
            do_task("del v2_Annotations_Val_mscoco.zip"),
            do_task("tar -xf v2_Questions_Train_mscoco.zip"),
            do_task("del v2_Questions_Train_mscoco.zip"),
            do_task("tar -xf v2_Questions_Val_mscoco.zip"),
            do_task("del v2_Questions_Val_mscoco.zip"),
        ]
    },
    "hmdb": {
        "link" : [
            "https://storage.googleapis.com/omninet/hmdb_data.zip"
        ],
        "post_download":[
            do_task("tar -xf hmdb_data.zip"),
            do_task("del hmdb_data.zip"),
            do_task("move hmdb/* ./"),
            do_task("rmdir hmdb")
        ]
    },
}
# Since penn data is paid data, download links are not included.

def setup_data():
    # assume data path
    data_dir = 'data'
    # check if data directory exists
    if not os.path.exists(data_dir):
        print_info('data/ directory does not exist')
        print_info('creating data/ directory')
        os.system(f'mkdir {data_dir}')

    # switch to data
    curdir = os.path.abspath('.')
    os.chdir(data_dir)

    # download the datasets
    for downloadable in data_downloadables.keys():
        # check if downloadable directory exists
        if not os.path.exists(f"{downloadable}"):
            print_info(f"{downloadable}/ directory does not exist")
            print_info(f"creating {downloadable}/ directory")
            os.mkdir(downloadable)
            # switch to downloadable directory
            os.chdir(downloadable)
            # check if link is a list or no
            if isinstance(data_downloadables[downloadable]['link'], list):
                # its a multipart download
                for dl_part in range(len(data_downloadables[downloadable]['link'])):
                    # download from link
                    if os.system(f"wget --no-check-certificate {data_downloadables[downloadable]['link'][dl_part]}"):
                        print_error(f"could not download part {dl_part} of dataset for {downloadable}")
                        exit(-1)
            else:
                # not multipart
                # download from link
                if os.system(f"wget {data_downloadables[downloadable]['link']}"):
                    print_error(f"could not download dataset for {downloadable}")
                    exit(-1)
            print_info(f"successfully downloaded dataset for {downloadable}")
            # execute post download steps
            for exec_step_num in range(len(data_downloadables[downloadable]['post_download'])):
                if data_downloadables[downloadable]['post_download'][exec_step_num]():
                    print_error(f"could not execute post download step '{exec_step_num}'")
                    exit(-1)
            # switch back up to data
            os.chdir('..')
        else:
            print_gen("warning", f"{downloadable}/ directory already exists")

    # switch back to root
    os.chdir(curdir)


if __name__ == '__main__':
    setup_data()
