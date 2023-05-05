import pygame
import random
import os
import sys
import time
import numpy as np
import torch

pygame.init()

class Minesweeper:
    def __init__(self,grid_width=10,grid_height=10,cell_size=50,mine_count=13,window=True):

        # 定义常量
        self.GRID_WIDTH = grid_width  # 游戏网格宽度
        self.GRID_HEIGHT = grid_height  # 游戏网格高度
        self.CELL_SIZE = cell_size  # 单元格尺寸
        self.MINE_COUNT = mine_count  # 雷的数量

        self.RED = (255, 0, 0)  # 炸弹颜色，红色
        self.WHITE = (255, 255, 255)  # 白色底色
        self.BLACK = (0, 0, 0)  # 黑色，表示未翻开的方块色
        self.GREY = (128, 128, 128)  # 灰色，表示翻开后的方块颜色

        self.font = pygame.font.SysFont(None, 30)  # 设置字体
        self.window = window
        self.akc=False
        if self.window:
            pygame.display.set_caption("Minesweeper")  # 设置窗口标题
            self.screen = pygame.display.set_mode((self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE))  # 设置可视化窗口大小

            self.r = 0.
            self.R = []
            self.actions = []
            self.condition = True
            self.map = np.zeros([self.GRID_WIDTH, self.GRID_HEIGHT])
            self.t = 0
            self.count = np.zeros([self.GRID_WIDTH, self.GRID_HEIGHT])

        else:
            self.r=0.
            self.R=[]
            self.actions=[]
            self.condition=True
            self.map=np.zeros([self.GRID_WIDTH,self.GRID_HEIGHT])
            self.t = 0
            self.count=np.zeros([self.GRID_WIDTH,self.GRID_HEIGHT])

        self.grid = [[0 for _ in range(self.GRID_HEIGHT)] for _ in range(self.GRID_WIDTH)]  # 创建二维数组，0表示无雷的安全区域
        self.mines = []  # 存储地雷的位置
        for i in range(self.MINE_COUNT):
            while True:
                x = random.randint(0, self.GRID_WIDTH - 1)  # 随机生成x坐标
                y = random.randint(0, self.GRID_HEIGHT - 1)  # 随机生成y坐标
                if (x, y) not in self.mines:  # 如果该位置没有地雷
                    self.mines.append((x, y))  # 将该位置添加到地雷列表中
                    self.grid[x][y] = -1  # 在该位置标记为地雷
                    break

        self.revealed = np.array([[False for _ in range(self.GRID_HEIGHT)] for _ in range(self.GRID_WIDTH)])  # 创建未翻开的方块信息
        if not self.window:
            self.status=self.get_status()
        else:
            self.status = self.get_status()

    def get_adjacent_cells(self, x, y):
        '''获取目标位置的相邻元素格'''
        cells = []
        for i in range(max(0, x - 1), min(x + 2, self.GRID_WIDTH)):
            for j in range(max(0, y - 1), min(y + 2, self.GRID_HEIGHT)):
                if i != x or j != y:
                    cells.append((i, j))
        return cells

    def get_status(self):
        '''获取环境当前状态'''
        status = (self.revealed.astype(np.float64) - 1) + self.map
        status = np.stack((status,self.count),axis=0)
        return status

    def reveal_cell(self, x, y):
        '''揭示指定位置的格子'''
        self.revealed[x][y] = True # 标记该位置已揭示
        if self.window:
            rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE) # 创建矩形
            pygame.draw.rect(self.screen, self.WHITE, rect) # 在该位置绘制白色矩形
            if self.grid[x][y] == -1: # 如果该位置是地雷
                self.map[x, y] = -10
                pygame.draw.circle(self.screen, self.RED, rect.center, self.CELL_SIZE // 3) # 在该位置绘制红色圆形
            else:
                pygame.draw.rect(self.screen, self.GREY, rect) # 在该位置绘制灰色矩形
                self.map[x, y] = self.count_adjacent_mines(x, y)
                if self.count_adjacent_mines(x, y) > 0: # 如果该位置周围有地雷
                    text = self.font.render(str(self.count_adjacent_mines(x, y)), True, self.BLACK) # 创建文本
                    text_rect = text.get_rect(center=rect.center) # 设置文本位置
                    self.screen.blit(text, text_rect) # 在该位置绘制文本
        else:
            if self.grid[x][y] == -1:
                self.map[x,y]=-10
            else:
                self.map[x,y]=self.count_adjacent_mines(x, y)

    def reveal_all_cells(self):
        '''将所有未揭示的格子都揭示出来'''
        for i in range(self.GRID_WIDTH):
            for j in range(self.GRID_HEIGHT):
                if not self.revealed[i][j]:
                    self.reveal_cell(i, j)

    def agent_click(self,x,y):
        '''智能体点击指定位置的格子'''
        if self.revealed[x][y]:
            # self.r+=-(100.+4*self.t)/100.
            self.r += 0.
        elif self.grid[x][y] != -1:
            self.reveal_cell(x,y)
            self.r=1.
            # self.r += (1.+2*self.revealed.sum()/(self.GRID_WIDTH*self.GRID_HEIGHT))
            if self.count_adjacent_mines(x, y) == 0: # 如果该位置周围没有地雷
                for i, j in self.get_adjacent_cells(x, y):
                    if self.grid[i][j] != -1 and not self.revealed[i][j]:
                        self.agent_click(i, j) # 递归揭示周围的位置
        else:
            self.reveal_all_cells()
            # self.r+=-10.
            self.r += 0.
            self.condition=False


    def handle_left_click(self, x, y):
        if self.revealed[x][y]:
            # self.r+=-(100.+4*self.t)/100.
            self.r += 0.
        elif self.grid[x][y] != -1: # 如果该位置不是地雷
            self.reveal_cell(x, y) # 揭示该位置
            self.r = 1.
            if self.count_adjacent_mines(x, y) == 0: # 如果该位置周围没有地雷
                for i, j in self.get_adjacent_cells(x, y):
                    if self.grid[i][j] != -1 and not self.revealed[i][j]:
                        self.handle_left_click(i, j) # 递归揭示周围的位置
        else:
            self.reveal_all_cells() # 揭示所有位置
            self.r += 0.
            self.condition = False
            pygame.display.flip()
            if self.akc:
                # self.running=False

                time.sleep(2.)
                self.reset()

    def handle_right_click(self, x, y):
        pass

    def draw_grid(self):
        for i in range(self.GRID_WIDTH):
            for j in range(self.GRID_HEIGHT):
                rect = pygame.Rect(i * self.CELL_SIZE, j * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE) # 创建矩形
                pygame.draw.rect(self.screen, self.WHITE, rect, 1) # 在该位置绘制白色矩形
                if self.revealed[i][j]: # 如果该位置已揭示
                    if self.grid[i][j] == -1: # 如果该位置是地雷
                        pygame.draw.circle(self.screen, self.RED, rect.center, self.CELL_SIZE // 3) # 在该位置绘制红色圆形
                    else:
                        pygame.draw.rect(self.screen, self.GREY, rect) # 在该位置绘制灰色矩形
                        if self.count_adjacent_mines(i, j) > 0: # 如果该位置周围有地雷
                            text = self.font.render(str(self.count_adjacent_mines(i, j)), True, self.BLACK) # 创建文本
                            text_rect = text.get_rect(center=rect.center) # 设置文本位置
                            self.screen.blit(text, text_rect) # 在该位置绘制文本

    def count_adjacent_mines(self, x, y):
        '''计数周围雷的数量'''
        count = 0
        for i, j in self.get_adjacent_cells(x, y):  # 循环变量周围的方块的坐标信息
            if self.grid[i][j] == -1:
                count += 1
        return count

    def reset(self):
        self.grid = [[0 for _ in range(self.GRID_HEIGHT)] for _ in range(self.GRID_WIDTH)]  # 创建二维数组，0表示无雷的安全区域
        self.mines = []  # 存储地雷的位置
        for i in range(self.MINE_COUNT):
            while True:
                x = random.randint(0, self.GRID_WIDTH - 1)  # 随机生成x坐标
                y = random.randint(0, self.GRID_HEIGHT - 1)  # 随机生成y坐标
                if (x, y) not in self.mines:  # 如果该位置没有地雷
                    self.mines.append((x, y))  # 将该位置添加到地雷列表中
                    self.grid[x][y] = -1  # 在该位置标记为地雷
                    break

        self.revealed = np.array(
            [[False for _ in range(self.GRID_HEIGHT)] for _ in range(self.GRID_WIDTH)])  # 创建未翻开的方块信息
        if not self.window:
            self.r = 0.
            self.R = []
            self.actions = []
            self.condition = True
            self.map = np.zeros([self.GRID_WIDTH, self.GRID_HEIGHT])
            self.status = self.get_status()
            self.t=0
            self.count=np.zeros([self.GRID_WIDTH,self.GRID_HEIGHT])
        else:
            self.r = 0.
            self.R = []
            self.actions = []
            self.condition = True
            self.map = np.zeros([self.GRID_WIDTH, self.GRID_HEIGHT])
            self.status = self.get_status()
            self.t = 0
            self.count = np.zeros([self.GRID_WIDTH, self.GRID_HEIGHT])

            # time.sleep(1.)
            self.screen = pygame.display.set_mode(
                (self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE))  # 设置可视化窗口大小
            self.screen.fill(self.BLACK)  # 填充黑色
            self.draw_grid()
            pygame.display.flip()  # 更新屏幕
            # time.sleep(1.)


    def update(self,a):
        [x,y]=a
        self.r=0.
        self.agent_click(x,y)
        self.count[x,y]+=1
        # if self.revealed.sum() == (self.GRID_WIDTH*self.GRID_HEIGHT-self.MINE_COUNT):
        #     self.r+=200.
        #     self.condition=False

        if self.revealed.sum() <=(self.GRID_WIDTH*self.GRID_HEIGHT-self.MINE_COUNT) and self.revealed.sum()>=(self.GRID_WIDTH*self.GRID_HEIGHT-self.MINE_COUNT-10) :
            self.r=50.
            self.condition=False

        self.status=self.get_status()
        self.R.append(self.r)
        self.actions.append([x,y])
        self.t+=1
        if self.t==50:
            self.condition=False
            # self.r=-20.
            self.r = 0.
        return [torch.tensor(self.status,dtype=torch.float32),self.r,self.condition]

    def agengt_run(self,a):
        [x,y]=a
        self.r = 0.
        self.handle_left_click(x,y)
        self.draw_grid()  # 绘制网格
        pygame.display.flip()  # 更新屏幕

        self.count[x, y] += 1
        if self.revealed.sum() <=(self.GRID_WIDTH*self.GRID_HEIGHT-self.MINE_COUNT) and self.revealed.sum()>=(self.GRID_WIDTH*self.GRID_HEIGHT-self.MINE_COUNT-10) :
            self.r=10.
            self.condition=False
        self.status = self.get_status()
        self.R.append(self.r)
        self.actions.append([x, y])
        self.t += 1
        if self.t==50:
            self.condition=False
            # self.r=-20.
            self.r = 0.
        return [torch.tensor(self.status,dtype=torch.float32),self.r,self.condition]

    def quit(self):
        # 退出pygame
        pygame.quit()

    def run(self):
        # 设置视频驱动为dummy
        # os.environ['SDL_VIDEODRIVER'] = 'dummy'
        self.akc=True

        # 主游戏循环
        self.running = True
        while self.running:
            # 处理事件
            for event in pygame.event.get():  # 当得到一个相应事件时
                if event.type == pygame.QUIT:  # 如果是退出事件（X按钮）
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:  # 如果是鼠标点击事件
                    x, y = event.pos # 得到鼠标在窗口中的位置
                    x //= self.CELL_SIZE  # 整除方块大小得到鼠标在具体哪个方块
                    y //= self.CELL_SIZE
                    if event.button == 1:
                        self.handle_left_click(x, y) # 处理左键点击事件
                    elif event.button == 3:
                        self.handle_right_click(x, y) # 处理右键点击事件

            # 绘制屏幕，刷新屏幕内容
            # self.screen.fill(self.BLACK) # 填充黑色
            self.draw_grid() # 绘制网格
            pygame.display.flip() # 更新屏幕

        # 退出pygame
        pygame.quit()
        # sys.exit()  # 退出Python程序

if __name__=='__main__':
    minesweeper = Minesweeper()
    minesweeper.run()
