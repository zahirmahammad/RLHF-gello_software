import sympy as sp
import numpy as np
import math

class ForwardKinematicsUR5e:
    """
    Class to calculate the forward kinematics of a 6-DOF robot arm.
    """

    def __init__(self):
        θdeg, αdeg, a, d = sp.symbols('θdeg αdeg a d')

    def Transformation_matrix(self, θdeg, αdeg, a, d):
        αrad = sp.rad(αdeg)
        θrad = sp.rad(θdeg)

        return sp.Matrix([
            [sp.cos(θrad), -sp.sin(θrad) * sp.cos(αrad), sp.sin(θrad) * sp.sin(αrad), a* sp.cos(θrad)],
            [sp.sin(θrad), sp.cos(θrad) * sp.cos(αrad), -sp.cos(θrad) * sp.sin(αrad), a* sp.sin(θrad)],
            [0, sp.sin(αrad), sp.cos(αrad), d],
            [0, 0, 0, 1]
        ])

    #Printing all the original transformation matrices with each joint at idle position UR5e
    def T1 (self, θdeg1):
        return self.Transformation_matrix(θdeg1, -90, 0, 162.5)
    def T2 (self, θdeg2):
        return self.Transformation_matrix((θdeg2 + 90), 180, -425, 0)
    def T3 (self, θdeg3):
        return self.Transformation_matrix(θdeg3, 0, -392.2, 0)
    def T4 (self, θdeg4):
        return self.Transformation_matrix((θdeg4 + 90), -90, 0, -133.3)
    def T5 (self,θdeg5):
        return self.Transformation_matrix(θdeg5,(-90), 0, 99.7)
    def T6 (self, θdeg6):
        return self.Transformation_matrix(θdeg6, 0, 0, 99.6)

    def get_transformation_matrix(self, θdeg1=0, θdeg2=0, θdeg3=0, θdeg4=0, θdeg5=0, θdeg6=0):
        # Multiplying the transformation matrices to obtain the final transformation matrix T
        T = self.T1(θdeg1) * self.T2(θdeg2) * self.T3(θdeg3) * self.T4(θdeg4) * self.T5(θdeg5) * self.T6(θdeg6)
        T = np.array(T.evalf()).astype(np.float64)
        # print("T", T.shape)
        return T

    def transform_to_rpy_and_xyz(self, T):
        """
        Extracts roll, pitch, yaw and x, y, z from a transformation matrix.

        Args:
            T (numpy.ndarray): A 4x4 transformation matrix.

        Returns:
            tuple: Roll, pitch, yaw, x, y, z.
        """
        assert T.shape == (4, 4), "Transformation matrix must be 4x4"

        # Extract the rotation matrix
        R = T[:3, :3]

        # Extract the translation vector
        x, y, z = T[:3, 3]

        # Calculate yaw (around z-axis)
        yaw = np.arctan2(R[1, 0], R[0, 0])

        # Calculate pitch (around y-axis)
        pitch = np.arcsin(-R[2, 0])

        # Calculate roll (around x-axis)
        roll = np.arctan2(R[2, 1], R[2, 2])

        return roll, pitch, yaw, x, y, z
    
    def convert_to_quaternions(self, roll, pitch, yaw):
        """
        Converts roll, pitch, yaw angles to quaternions.

        Args:
            roll (float): Roll angle in radians.
            pitch (float): Pitch angle in radians.
            yaw (float): Yaw angle in radians.

        Returns:
            tuple: Quaternion (w, x, y, z).
        """
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return w, x, y, z

    # Example usage:
    # roll, pitch, yaw = 0.1, 0.2, 0.3
    # w, x, y, z = convert_to_quaternions(roll, pitch, yaw)
    # print(f"Quaternion: ({w:.2f}, {x:.2f}, {y:.2f}, {z:.2f})")


# fk = ForwardKinematicsUR5e()

# # Now printing the positions rotated by 90 deg individually and obtaining 5 geometrically known configurations by rotating each joint by 90 degrees
# print ("Now printing the positions rotated by 90 deg individually and obtaining 5 geometrically known configurations by rotating each joint by 90 degrees")
# print ("For θ1 = 0 we get the transformation matrix of end eff. wrt 0 as")
# T = fk.get_transformation_matrix()
# sp.pprint(T)

# T = np.array(T).astype(np.float64)

# roll, pitch, yaw, x, y, z = fk.transform_to_rpy_and_xyz(T)

# print(f"Roll: {roll:.2f} radians, Pitch: {pitch:.2f} radians, Yaw: {yaw:.2f} radians")
# print(f"x: {x}, y: {y}, z: {z}")

# w, x, y, z = fk.convert_to_quaternions(roll, pitch, yaw)

# print(f"Quaternion: ({w:.2f}, {x:.2f}, {y:.2f}, {z:.2f})")

    #Printing all the original transformation matrices with each joint at idle position UR10
    # def T1 (θdeg1):
    #     return Transformation_matrix(θdeg1, -90, 0, 127.3)
    # def T2 (θdeg2):
    #     return Transformation_matrix((θdeg2 + 90), 180, -612.7, 0)
    # def T3 (θdeg3):
    #     return Transformation_matrix(θdeg3, 0, -572.3, 0)
    # def T4 (θdeg4):
    #     return Transformation_matrix((θdeg4 + 90), -90, 0, -163.941)
    # def T5 (θdeg5):
    #     return Transformation_matrix(θdeg5,(-90), 0, 115.7)
    # def T6 (θdeg6):
    #     return Transformation_matrix(θdeg6, 0, 0, 92.2)