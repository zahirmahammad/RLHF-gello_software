# GELLO
This is the central repo that holds the all the software for GELLO. See the website for the paper and other resources for GELLO https://wuphilipp.github.io/gello_site/
See the GELLO hardware repo for the STL files and hardware instructions for building your own GELLO https://github.com/wuphilipp/gello_mechanical
```
git clone https://github.com/wuphilipp/gello_software.git
cd gello_software
```

### Instructions

1. Download and Clone the repository
2. Manually install the libraries and dependencies from the requriements.txt file or from the github link above
3. Open the folder /gello_software in the terminal
4. Run the python file to fire up mujoco
   ```
   python experiments/launch_nodes.py --robot sim_urspoon
   ```
5. Connect the Gello hardware and run the python script
   ```
   python experiments/run_env.py --agent=gello
   ```
6. Additionally, joint angles can also be passed from the csv file, Run the python script below to execute one of the joint angle trajectories
   ```
   python experiments/joint_angles.py
   ```

   
