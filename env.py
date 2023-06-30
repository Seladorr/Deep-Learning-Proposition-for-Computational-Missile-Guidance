import numpy as np
import random
import math

class env():
    def __init__(self,initial, v_t, v_m, a_max, Ta = 0.01):
        self.Ta = Ta
        self.r = initial[0]
        self.lamda = initial[1]*math.pi/180
        self.gamma_m = initial[2]*math.pi/180
        self.gamma_t = initial[3]*math.pi/180
        self.a_max = a_max
        self.a_m = 0

        self.done = False

        self.v_t = v_t
        self.v_m = v_m
        self.score = 0

        self.r_dot = self.v_t*math.cos(initial[3] - initial[1]) - self.v_m*math.cos((initial[2] - initial[1]))
        self.lamda_dot = (self.v_t*math.sin(initial[3] - initial[1]) - self.v_m*math.sin((initial[2] - initial[1])))/self.r

        self.r_dot_initial = self.v_t*math.cos(initial[3] - initial[1]) - self.v_m*math.cos((initial[2] - initial[1]))
        self.lamda_dot_initial = (self.v_t*math.sin(initial[3] - initial[1]) - self.v_m*math.sin(initial[2] - initial[1]))/self.r


    def dynamics(self,a_g):
        a_g = a_g*self.a_max
        # print('a_m:',a_m)
        self.a_m = self.a_m+((a_g-self.a_m)/0.2)*self.Ta
        if(self.a_m<-400):
            self.a_m = -400
        if(self.a_m>400):
            self.a_m = 400
        a_t = 0
        self.r_dot = self.v_t*math.cos(self.gamma_t - self.lamda) - self.v_m*math.cos(self.gamma_m - self.lamda)
        self.r = self.Ta*(self.r_dot) + self.r
        self.lamda_dot = (self.v_t*math.sin(self.gamma_t - self.lamda) - self.v_m*math.sin(self.gamma_m - self.lamda))/self.r
        self.lamda = self.Ta*(self.lamda_dot)+self.lamda    
        self.gamma_m = self.Ta*(self.a_m/self.v_m) + self.gamma_m
        self.gamma_t = self.Ta*(a_t/self.v_t) + self.gamma_t

        
        self.new_state = [self.r_dot/self.r_dot_initial, self.lamda_dot/self.lamda_dot_initial]
        if(self.r<=1):
            self.done = True
        else:
            self.done = False
        return self.new_state, self.reward(), self.r, self.done

    def reward(self):
        k_a = -0.2
        k_r = -2.0
        k_dr = -2.0
        k_ter = 10.0
        r_max = 1.0
        
        zem = self.r/math.sqrt(1+math.pow(self.r_dot/(self.r*self.lamda_dot),2))
        
        R_a = k_a*math.pow(self.a_m/self.a_max,2)
        R_r = k_r*math.pow(zem/5000.0,2)
        
        if(self.r_dot>0):
            R_dr = k_dr
        else:
            R_dr = 0
        if(self.r<=r_max):
            R_ter = k_ter
        else:
            R_ter = 0
        reward = R_a + R_r + R_dr + R_ter
        self.score = self.score + reward
        # print('reward:',reward)
        return reward

    

        





