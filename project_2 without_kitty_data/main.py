# Mapping and perception for autonomous robot , semester A 2021/22
# Roy orfaig & Ben-Zion Bobrovsky
# Project 2
# KF, EKF and EKF-SLAM! 

import os
from data_loader import DataLoader
from project_questions import ProjectQuestions


if __name__ == "__main__":
    basedir = 'C:\project2'#example
    date = '2011_09_30' #example (fill your correct data)
    drive = '0033' #The recording number I used in the sample in class  (fill your correct data)
    dat_dir = os.path.join(basedir,"Ex3_data")
    
    dataset = DataLoader(basedir, date, drive, dat_dir)
    
    project = ProjectQuestions(dataset)
    project.run()