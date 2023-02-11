import os
import shutil
import glob
import urllib

import requests
import torch
import json
from lxml import etree
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def setup_dataset(name):
    creds = Credentials.from_authorized_user_file('../client_secret.json')
    parent_folder = []
    sub_folders = []

    try:
        # create drive api client
        service = build('drive', 'v3', credentials=creds)
        page_token = None
        while True:
            response = service.files().list(q="mimeType='application/vnd.google-apps.folder' and name='ML Datasets'",
                                            spaces='drive',
                                            fields='nextPageToken, '
                                                   'files(id, name)',
                                            pageToken=page_token).execute()
            for file in response.get('files', []):
                # Process change
                print(F'Found {file.get("name")}')
            parent_folder.extend(response.get('files', []))

            if parent_folder:
                parent_folder_id = parent_folder[0]['id']
                response = service.files().list(
                    q="mimeType='application/vnd.google-apps.folder' and parents in '" + parent_folder_id + "'",
                    fields="nextPageToken, files(id, name)").execute()
                for file in response.get('files', []):
                    # Process change
                    print(F'{file.get("name")}, {file.get("id")}')
                sub_folders.extend(response.get('files', []))
                page_token = response.get('nextPageToken', None)
                if page_token is None:
                    break

        dataset_obj = []
        for file in sub_folders:
            if file['name'] == name:
                dataset_obj = file

        if not dataset_obj:
            print(f"No dataset with name '{name}' found.")
            return

        if not os.path.exists(name):
            os.mkdir(name)
        else:
            print(f"Folder with name '{name}' exists already.")
            return

        download_dataset(service, dataset_obj['id'], name)

    except HttpError as error:
        print(F'An error occurred: {error}')
        parent_folder = None


def download_dataset(drive_service, folder_id, target_dir, next_page_token=None):
    try:
        # Define the query parameters
        query = "trashed = false and parents in '" + folder_id + "'"
        if next_page_token:
            query += " and pageToken='" + next_page_token + "'"

        # Search for folders
        response = drive_service.files().list(q=query, fields="nextPageToken, files(id, name, mimeType)").execute()
        items = response.get("files", [])

        for item in items:
            # Recursively call the function for each subfolder
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                download_dataset(drive_service, item['id'], target_dir + '/' + item['name'])
            else:
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                    print(f"Directory '{target_dir}' made")
                # Download the file
                file = drive_service.files().get_media(fileId=item['id']).execute()
                with open(target_dir + '/' + item['name'], "wb") as f:
                    f.write(file)
                    print(f"File with name '{item['name']}' has been downloaded.")

        # Check if there's a next page token
        next_page_token = response.get("nextPageToken", None)
        if next_page_token:
            download_dataset(drive_service, folder_id, target_dir, next_page_token)

    except HttpError as error:
        print(f'An error occurred: {error}')
        print('The download process was interrupted.')


train_dir = '.\\dataset\\train'
test_dir = '.\\dataset\\test'
val_dir = '.\\dataset\\val'


def check_folders():
    if 'train' not in os.listdir():
        print('Creating train directory')
        os.mkdir(train_dir)
    if 'test' not in os.listdir():
        print('Creating test directory')
        os.mkdir(test_dir)
    if 'val' not in os.listdir():
        print('Creating val directory')
        os.mkdir(val_dir)


def split_train_and_val(train_val_dir, dest_ext_str):
    files = glob.glob(train_val_dir + '\\*')
    files = sorted(files, key=lambda x: int(os.path.basename(x).split(".")[0]))
    assert len(files) != 0, f'No files found in this folder'

    filecount = len(os.listdir(train_val_dir))
    train_filecount = int(filecount * 0.8)
    val_filecount = int(filecount * 0.2)

    # Extension to destination path e.g. Images or Labels
    dest_train_dir = os.path.join(train_dir, dest_ext_str)
    dest_val_dir = os.path.join(val_dir, dest_ext_str)

    try:
        os.mkdir(dest_train_dir)
        os.mkdir(dest_val_dir)
    except FileExistsError:
        print('Directory(ies) already exist')

    # Copy 80% of all files to training
    for file in files[:train_filecount]:
        print('Moving', file)
        shutil.move(file, dest_train_dir)

    # Copy ensured 20% of all files to validation
    files = glob.glob(train_val_dir + '\\*')
    files = sorted(files, key=lambda x: int(os.path.basename(x).split(".")[0]))
    assert len(files) != 0, f'No files found in this folder'
    if len(files) == val_filecount:
        for file in files:
            print('Moving', file)
            shutil.move(file, dest_val_dir)


def convert_to_yolo_label(xml_label_path):
    files = glob.glob(xml_label_path + '\\*.xml')
    assert len(files) != 0, f'No files found in this folder'

    for file in files:
        # parse the xml file
        xml_root = etree.parse(file)

        # extract the width and height of the image
        img_width = int(xml_root.xpath("/annotation/size/width/text()")[0])
        img_height = int(xml_root.xpath("/annotation/size/height/text()")[0])

        file = file[:file.find(".xml")] + ".txt"
        print('Writing', file)

        # open the text file to write the labels
        with open(file, "w") as f:
            # iterate through all objects in the xml file
            for obj in xml_root.xpath("/annotation/object"):
                # extract the bounding box coordinates
                xmin = int(obj.xpath("bndbox/xmin/text()")[0])
                ymin = int(obj.xpath("bndbox/ymin/text()")[0])
                xmax = int(obj.xpath("bndbox/xmax/text()")[0])
                ymax = int(obj.xpath("bndbox/ymax/text()")[0])

                # calculate the center coordinates and width/height of the bounding box
                x_center = (xmin + xmax) / (2 * img_width)
                y_center = (ymin + ymax) / (2 * img_height)
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                # write the label to the text file
                f.write("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(x_center, y_center, width, height))


def purge_xml_labels(xml_label_path):
    files = glob.glob(xml_label_path + '\\*.xml')
    assert len(files) != 0, f'No files found in this folder'
    for file in files:
        print('Removing', file)
        os.remove(file)


def prepare_data():
    try:
        check_folders()
    except FileExistsError:
        print('Directory already exists')
    try:
        split_train_and_val('.\\Train-Val', 'images')
        split_train_and_val('.\\Labelled Train-Val', 'labels')

        convert_to_yolo_label('.\\dataset\\train\\labels')
        purge_xml_labels('.\\dataset\\train\\labels')

        convert_to_yolo_label('.\\dataset\\val\\labels')
        purge_xml_labels('.\\dataset\\val\\labels')

        convert_to_yolo_label('.\\dataset\\test\\labels')
        purge_xml_labels('.\\dataset\\test\\labels')
    except AssertionError:
        print('No files found')


def main():
    setup_dataset('BSVSO')


if __name__ == '__main__':
    main()
