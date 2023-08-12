import pygame
import random 
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from random import randint

#window size
WIDTH = 360
HEIGHT = 360
FPS = 30#how fast game is
#colors
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
GREY = (128,128,128)
YELLOW = (255,255,0)

class Player(pygame.sprite.Sprite):
     #sprite for the player
     def __init__(self):
         pygame.sprite.Sprite.__init__(self)
         self.image = pygame.Surface((20,20))
         self.image.fill(BLUE)
         self.rect = self.image.get_rect()
         self.rect.centerx = WIDTH/2
         self.rect.bottom = HEIGHT-10
         self.speedx= 0
         self.speedy = 0
     def update(self,action):
         self.speedx = 0
         self.speedy = 0
         keystate = pygame.key.get_pressed()
         
         if keystate[pygame.K_LEFT] or action == 0:
             self.speedx = -4
         elif keystate[pygame.K_RIGHT] or action == 1:
             self.speedx = 4
         else:
             self.speedx = 0
         
            
         self.rect.x +=self.speedx
         self.rect.y +=self.speedy
         if self.rect.right > WIDTH:
             self.rect.right = WIDTH
         if self.rect.bottom > HEIGHT:
             self.rect.bottom = HEIGHT
         if self.rect.left < 0:
             self.rect.left = 0
         if self.rect.top < 0:
             self.rect.top = 0
        
       
       
       
     def getCoordinates(self):
        return (self.rect.x, self.rect.y)
             
class Golden(pygame.sprite.Sprite):
    
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((25,25))
        self.image.fill(YELLOW)
        self.rect = self.image.get_rect()
        
        self.rect.x = random.randrange(0, WIDTH - self.rect.width)
        self.rect.y = random.randrange(2,6)
               
        self.speedx = 0
        self.speedy = 3
    
    def update(self):
         self.rect.x += self.speedx
         self.rect.y += self.speedy
         
         if self.rect.top > HEIGHT + 10:
             self.rect.x = random.randrange(0, WIDTH - self.rect.width)
             self.rect.y = random.randrange(2,6)
             self.speedy = 3
             
             
    def getCoordinates(self):
        return (self.rect.x, self.rect.y)
 
class Silver(pygame.sprite.Sprite):
    
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((25,25))
        self.image.fill(GREY)
        self.rect = self.image.get_rect()
        
        self.rect.x = random.randrange(0, WIDTH - self.rect.width)
        self.rect.y = random.randrange(2,6)
               
        self.speedx = 0
        self.speedy = 3
    
    def update(self):
         self.rect.x += self.speedx
         self.rect.y += self.speedy
         
         if self.rect.top > HEIGHT + 10:
             self.rect.x = random.randrange(0, WIDTH - self.rect.width)
             self.rect.y = random.randrange(2,6)
             self.speedy = 3
             
             
    def getCoordinates(self):
        return (self.rect.x, self.rect.y)      

class DQLAgent:
    def __init__(self):
        #parameter / hyperparameter
        self.state_size = 4 # player ve enemyler arasındaki x ler ve y ler arası distanceler
        self.action_size = 3 #sağ sol ya da yerinde kal
        
        self.gamma = 0.95
        self.learning_rate = 0.01
        self.epsilon = 1 #explore
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.memory = deque(maxlen = 1000)
        
        self.model = self. build_model()

    def build_model(self):
        #neural network for deep q learning
        model = Sequential()
        model.add(Dense(48, input_dim = self.state_size, activation = "relu"))
        model.add(Dense(self.action_size, activation = "linear"))
        model.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        #storage
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self,state):
        state = np.array(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)#aslında q value
        return np.argmax(act_values[0])
        
    def replay(self, batch_size):
        #training
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory,batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.array(state)
            next_state = np.array(next_state)
            if done:
                target = reward
            else:
                target = reward + self.gamma*np.max(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state,train_target, verbose = 0)
            
    def adaptiveGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    
class Env(pygame.sprite.Sprite):
    def __init__(self):
        
        pygame.sprite.Sprite.__init__(self)
        self.all_sprite = pygame.sprite.Group()
        self.enemy = pygame.sprite.Group()
        self.player = Player()
        self.all_sprite.add(self.player)
        self.m1 = Golden()
        self.m2 = Silver()
        self.m3 = None
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)
        self.enemy.add(self.m1)
        self.enemy.add(self.m2)
        
        self.reward = 0
        self.done = False
        self.total_reward = 0
        self.agent = DQLAgent()
    
    def findDistance(self, a, b):
        distance = a-b
        return distance
    
    def step(self, action, reward):
        state_list = []
        
        #update
        self.player.update(action)
        self.reward -= 1
        self.enemy.update()
        
        #get coordinate
        next_player_state = self.player.getCoordinates()
        next_m1_state = self.m1.getCoordinates()
        next_m2_state = self.m2.getCoordinates()
        
        #find distance
        state_list.append(self.findDistance(next_player_state[0],next_m1_state[0]))
        state_list.append(self.findDistance(next_player_state[1],next_m1_state[1]))
        state_list.append(self.findDistance(next_player_state[0],next_m2_state[0]))
        state_list.append(self.findDistance(next_player_state[1],next_m2_state[1]))
        
        return [state_list]
    #reset
 
    def initialState(self):

                  
            
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)
        self.enemy.add(self.m1)
        self.enemy.add(self.m2)
        self.m1 = Golden()
        self.m2 = Silver()
        self.reward = 0
        self.done = False
        self.total_reward = 0     
        
        state_list = []
        #getCoordinate
        player_state = self.player.getCoordinates()
        m1_state = self.m1.getCoordinates()
        m2_state = self.m2.getCoordinates()    
        
        state_list.append(self.findDistance(player_state[0],m1_state[0]))
        state_list.append(self.findDistance(player_state[1],m1_state[1]))
        state_list.append(self.findDistance(player_state[0],m2_state[0]))
        state_list.append(self.findDistance(player_state[1],m2_state[1]))
        
        return [state_list]        
    
    def NextEpisode(self):
        
        
        pygame.sprite.Sprite.__init__(self)
        self.all_sprite = pygame.sprite.Group()
        self.enemy = pygame.sprite.Group()
        self.player = Player()
        self.all_sprite.add(self.player)
        
        liste = [0,0,0,0,0,1,1,1,1,1]


        x = randint(1,2)
        print ("x:", x)
        def BlaBla():
            meyveler = []            
            i = 0    
            while i < x:
                
                #        print ("i : ", i)
                F = randint(0,9)
                print ("hiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
                F1 = liste[F]
                print ("F1 : " , F1)
                meyveler.append(str(F1))
                i += 1
            return meyveler
        
        
        meyveler = BlaBla()

        pygame.sprite.Sprite.__init__(self)
        self.all_sprite = pygame.sprite.Group()
        self.enemy = pygame.sprite.Group()
        self.player = Player()
        self.all_sprite.add(self.player)
        print ("bbb:", len(meyveler))
        if len(meyveler) == 2:
            print (meyveler[0])
            print (meyveler[1])
            if meyveler[0] == "0" and meyveler[1] == "0":
                self.m1 = Golden()
                self.m2 = Golden()
            elif meyveler[0] == "0" and meyveler[1] == "1":
                self.m1 = Golden()
                self.m2 = Silver()
            elif meyveler[0] == "1" and meyveler[1] == "0":
                self.m1 = Silver()
                self.m2 = Golden()
            else:
                self.m1 = Silver()
                self.m2 = Silver()        
        else:
            print (meyveler[0])
            
            if meyveler[0] == "0":
                self.m1 = Golden()
                self.m2 = Golden()
            elif meyveler[0] == "1":
                self.m1 = Silver()
                self.m2 = Silver()        
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)
        self.enemy.add(self.m1)
        self.enemy.add(self.m2)        
        
        
        
        
    def run(self):
        #game loop
        state = self.initialState()
        running = True
        batch_size = 24
        t = 0
        while running:
            self.reward = self.reward
            #keep loop running at the right speed
            clock.tick(FPS) 
            
                #process input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            #update
            action = self.agent.act(state)
            next_state = self.step(action,self.reward)
            self.total_reward += self.reward
            
            hits = pygame.sprite.spritecollide(self.player, self.enemy, False, pygame.sprite.collide_circle) 
            Furkan = hits.copy()
            
            if hits:
                t += 1
                print ("Episode:", t)
                print ("aaaa:", str(Furkan[0])[1:7])
                if str(Furkan[0])[1:7] == "Golden" :
                    
                    
                
#                print ("selfenemy:", self.enemy)
                    self.NextEpisode()
                    self.reward += 1000000
                    self.total_reward += self.reward                
                    self.done = True
                    self.reward = 0 
                    #running = True
                    print("Total Reward:", self.total_reward)
                    
                else:
                    self.NextEpisode()
                    self.reward += 1
                    self.total_reward += self.reward                
                    self.done = True
                    self.reward = 0 
                    #running = True
                    print("Total Reward:", self.total_reward)               
                    
            #storage
            self.agent.remember(state, action,self.reward, next_state, self.done)#burada hit olmadıysa devamında ne olacağı belirtiliyor, benimkinde hit olduktan sonra bu remember metodunu çağırıcaz!!!                    
           
#   kontrol için remembera atarken false atsın istiyoruz         self.done = False
            
            
            #update state
            state = next_state
            #training
            self.agent.replay(batch_size)
            
            #epsilon Greedy
            self.agent.adaptiveGreedy()#sonraki episode da ne yapacğaıma karar verdiğim yer
            
            
            #draw and show(render)        
            screen.fill(GREEN)
            self.all_sprite.draw(screen)
            #after drawing flip the display
            pygame.display.flip()
            
        pygame.quit() 


if __name__ == "__main__":
    env = Env()
#    lis = []
#    t = 0
    while True:
#        t += 1
#        print ("Episode:", t)
#        lis.append(env.total_reward)
           
        #initialize pygame and create window
        pygame.init()
        screen = pygame.display.set_mode((WIDTH,HEIGHT))
        pygame.display.set_caption("COLLECT GAME")
        clock = pygame.time.Clock()
        
        env.run()



#liste = [m1,m1,m1,m1,m1,m2,m2,m2,m2,m2]
#    meyveler = []   
#    x = np.random(0,2) 
#    for i in range x:
#        F = np.random(0,10)
#        F = liste [F]
#        meyveler.append(F)























    
            