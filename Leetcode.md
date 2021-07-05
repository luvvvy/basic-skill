## 二分法
1. 两数之和
12.LCP. 小张刷题计划
33. 搜索旋转排序数组
81. 搜索旋转排序数组II
1552. 两球之间的磁力
5678. 袋子里数目最少的球


## 双指针
1. 两数之和
9. 回文数
11. 盛水最多的容器
15. 三数之和
26. 删除排序数组中的重复项 （清爽的双指针）（C语言中删除重复项经典！COMP9315）
27. 删除元素 （清爽的双指针）
80. 删除排序数组中的重复项 II
283. 移动零
344. 反转字符串
922. 按奇偶排序数组 II

## 回文
125. 验证回文串
131. 分割回文串
132. 分割回文串II
214. 最短回文串
516. 最长回文子序列
647. 回文子串

## 栈、队列
20. 有效的括号
32. 最长的有效括号
121. 买股票的最佳时期
150. 逆波兰表达式求值
155. 最小栈
394. 字符串解码
503. 下一个更大元素II
735. 行星碰撞
739. 每日温度
1006. 笨阶乘
1047. 删除字符串中的所有相邻重复项
1249. 移除无效的括号

## 链表
21. 合并两个有序链表

22. 剑指offer.反转链表

    ```
    # -*- coding:utf-8 -*-
    # class ListNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.next = None
    class Solution:
        # 返回ListNode
        def ReverseList(self, pHead):
            nex_head = None
            pre_head = None
            cur = pHead
            while cur:
                #先用p指针记录当前结点的下一个结点地址。
                temp = cur.next
                #让被当前结点与链表断开并指向前一个结点pre。
                cur.next = pre_head
                #pre指针指向当前结点
                pre_head = cur
                #head指向p(保存着原链表中head的下一个结点地址)
                cur = temp
            return pre_head
    ```

    

23. 旋转链表

24. 删除排序链表中的重复元素

25. 环形链表

## 树
###  1.剑指 Offer 37. 序列化二叉树
	你需要设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构

#### 解题:

##### 1.什么是class，什么是instance，什么是object？什么是method，什么是function？什么是self?

1. 什么是class:理解为一个组装工厂,然后里面需要一个__init__就是初始化方法,来定义含有参数和变量,但是这时候没有任何在跑得实例,接下来 我们就要来创建实例.
 ```python3
        class BuildRobot():
            def __init__(self,armcount,headcount):
                self.armcount = armcount
                self.headcount = headcount
 ```

2. instance就是实例及运行的机器人,这和object几乎等价
 ```python
        normal_robot = BuildRobot(2,1)
        normal_robot.armcount
 ```
3. 在class里面的function叫method。所以，method是和class，instance有关的一种function
 - 如果我要在class里写方法, 这个方法想要使用必须先创建instance,
- 相反如果我不在class里写def,那么就不需要instance
  4. self， 就是指由这个class造出来的instance
- self， 就是指由这个class造出来的instance嘛。
  
- 在使用class里的method时候,如果在class里 我们么有创建实例,所以无法确认调用的instance,这时候就要创建一个self来代替未来的实例,所以
  
- self是在为class编写instance method的时候，放在变量名第一个位子的占位词。
  
- 在具体编写instance method里，可以不使用self这个变量。

- 如果在method里面要改变instance的属性，可以用self.xxxx来指代这个属性进行修改。
##### 2.解题思路:

1. 方法一:深度优先算法:
- 遍历树来,一般有两个策略：广度优先搜索和深度优先搜索。

广度优先搜索可以按照层次的顺序从上到下遍历所有的节点,及层序遍历
深度优先搜索可以从一个根开始，一直延伸到某个叶，然后回到根，到达另一个分支。根据根节点、左节点和右节点之间的相对顺序，可以进一步将深度优先搜索策略区分为：
先序遍历,中序遍历,后序遍历 本题采用先序遍历的方法
    
 ```python3
         class Codec:
           def serialize(self, root):
               if not root : return "None,"
               res = str(root.val) +"," 
               res += self.serialize(root.left)
               res += self.serialize(root.right)
               return res
         
             def deserialize(self, data):
               data = iter(data.split(','))
                 def des():
                   index = next(data)
                     if index=="None": return None
                   tree_a = TreeNode(index)
                   tree_a.val = index
                   tree_a.left = des()
                   tree_a.right = des()
                   return tree_a
                 return des()
         #next() 返回迭代器的下一个项目。
         #next() 函数要和生成迭代器的 iter() 函数一起使用。
 ```
2. **方法二: 层序遍历(广度优先)**
   
           class Codec:
               def serialize(self, root):
                   if not root: return ""
                   res = ""
                   queue = [root]
                   while queue:
                       root = queue.pop(0)
                       if not root:
                           res += "None,"
                           continue
                       res += str(root.val) + ","
                       queue.append(root.left)
                       queue.append(root.right)
                   return res
           
               def deserialize(self, data):
                   if not data: return None
                   data = iter(data.split(","))
                   res = TreeNode(next(data))
                   queue = [res]
                   while queue:
                       root = queue.pop(0)
                       left, right = next(data), next(data)
                       if left != "None":
                           root.left = TreeNode(left)
                           queue.append(root.left)
                       if right != "None":
                           root.right = TreeNode(right)
                           queue.append(root.right)
                   return res

### 2. 386. 字典序排数
	给定 n =1 3，返回 [1,10,11,12,13,2,3,4,5,6,7,8,9] 。
	请尽可能的优化算法的时间复杂度和空间复杂度。 输入的数据 n 小于等于 5,000,000。

- 使用深度优先算法加递归
```python3
class Solution(object):
    def lexicalOrder(self, n):
        lists = []
        for i in range(1,10):
            dfs(n,i,lists)
        return lists

def dfs( n,i,lists):
    if i>n: return None
    lists.append(i)
    for j in range(10):
        dfs(n,i*10+j,lists)
    return lists
```

### 3. 二叉树的最近公共祖先
- 写遍历,注意遍历条件
```
  class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        if not root: return
        if root==p or root ==q:
            return root
        L_res = self.lowestCommonAncestor(root.left,p,q) 
        R_res = self.lowestCommonAncestor(root.right,p,q)
        if L_res and R_res:
            return root
        elif L_res and not R_res: 
            return L_res
        elif not L_res and  R_res: 
            return R_res
```

### 4. 剑指offer.最小的K个数（最大堆实现）

```
方法1:排序法


```

#### 5.求深度

```python3
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
 
#
# 
# @param root TreeNode类 
# @return int整型
#
class Solution:
    def maxDepth(self , root ):
        # write code here,
        """
        # 递归
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
        """
        """
        # 层序遍历
        if not root:
            return 0
        queue = [root]
        res = 0
        while queue:
            temp = queue
            for node in temp:
                queue = queue[1:]
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res += 1
        return res
         """
         
        if not root:
            return 0
        queue = [root]
        temp = 1
        level = [temp]
        while queue:
            node = queue[0]
            queue = queue[1:]
            temp = level[0]
            level = level[1:]
            if node.left:
                queue.append(node.left)
                level.append(temp+1)
            if  node.right:
                queue.append(node.right)
                level.append(temp+1)
        return temp
```

### 6. [求根节点到叶节点数字之和](https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/)

```
前序遍历:
class Solution(object):
    def sumNumbers(self, root):
        def dfs(root, num):
            if not root :return 0
            
            if not root.left and not root.right:
                return 10*num + root.val
            else: 
                return dfs(root.left,10*num + root.val) + dfs(root.right,10*num + root.val)
        return dfs(root,0)
层序遍历:
class Solution(object):
    def sumNumbers(self, root):
        if not root: return 0
        queue = [root]
        queue_2 = [root.val]
        sum = 0
        while queue:
            temp = queue.pop(0)
            temp_val = queue_2.pop(0)
            if not temp.right and not temp.left:
                sum += temp_val
            else:
                if temp.left:
                    queue.append(temp.left)
                    queue_2.append(temp_val*10 + temp.left.val)
                if temp.right:
                    queue.append(temp.right)
                    queue_2.append(temp_val*10 + temp.right.val)
        return sum

```

### 7.二叉树遍历

```
使用栈的方法:
class Solution(object):
	def inorderTraversal(self, root):
		res = []
		stack = []
		while stack or root:
			# 不断往左子树方向走，每走一次就将当前节点保存到栈中
			# 这是模拟递归的调用
			if root:
				stack.append(root)
				root = root.left
			# 当前节点为空，说明左边走到头了，从栈中弹出节点并保存
			# 然后转向右边节点，继续上面整个过程
			else:
				tmp = stack.pop()
				res.append(tmp.val)
				root = tmp.right
		return res
```



### 4. 二叉树的中序遍历 （最小堆）
### 5. 相同的树
### 6. 路径总和
### 7. 数组中的第K个最大元素
### 8. 反转二叉树
### 9. 二叉树的所有路径
### 10. 另一个树的子树
### 11. 数据流中的第K大元素 
### 12. 二叉搜索树的范围和
### 13. 叶子相似的树


## hash
1. 两数之和
3. 剑指offer.数组中重复的数字
49. 字母异位词分组
205. 同构字符串
383. 赎金信

## 滑动窗口
3. 无重复字符的最长字串
459. 重复的子字符串（滑动窗口思想）
480. 滑动窗口的中位数
567. 字符串的排列
643. 子数组最大平均数
1004. 最大连续1的个数III
1052. 爱生气的书店老板
1208. 尽可能使字符串相等
1423. 可获得的最大点数


## DFS、BFS、回溯
16. 19. 面试题.  水域大小
17. 电话号码的字母组合
38. 剑指offer. 字符串的排列
39. 组合总和
46. 全排列
54. 螺旋矩阵
77. 组合（剪枝经典例子）
78. 子集
79. 单词搜索 
131. 分割回文串
200. 岛屿数量 （DFS）
419. 甲板上的战舰
463. 岛屿的周长
491. 递增子序列（剪枝经典例子）
547. 省份数量
695. 岛屿最大面积 （DFS）
765. 情侣牵手（BFS）

## 贪心算法
45. 跳跃游戏 II
55. 跳跃游戏
435. 无重叠区间
452. 最少的箭射爆气球
646. 最长数对序列
5673. 移除石子的最大得分

## 动态规划
5. 最长回文子串
    8.01.面试题.三步问题

6. 接雨水

7. 最大子序和

8. 跳跃游戏

### 9. 不同路径

   ```
   shopee的办公室非常大，小虾同学的位置坐落在右上角，而大门却在左下角，可以把所有位置抽象为一个网格（门口的坐标为0，0），小虾同学很聪明，每次只向上，或者向右走，因为这样最容易接近目的地，但是小虾同学不想让自己的boss们看到自己经常在他们面前出没，或者迟到被发现。他决定研究一下如果他不通过boss们的位置，他可以有多少种走法？
   使用动态规划:
   初步分析,当前走法是由前一步,也就是当前往左或者当前往下实现的.所以可以画出表格所有位置设置为1然后依次去叠加
   
   x,y,n = map(int,input().split())
   dpc = [[1 for j in range(y+1)]for j in range(x+1)]
   for _ in range(n):
       i,j = map(int,input().split())
       dpc[i][j] = 0
   for i in range(1,x+1):
       for j in range(1,y+1):
           if dpc[i][j]!=0:
               dpc[i][j] = dpc[i-1][j]+dpc[i][j-1]
   print(dpc[-1][-1])
   
   ```



10. 最小路径和

11. 分割回文串II

12. 打家劫舍

13. 最大正方形

14. 最长上升子序列

15. 零钱兑换

16. 分割等和子集

17. 无重叠区间

18. 最长回文子序列

19. 回文子串

20. 使用最小花费爬楼梯

21. 不相交的线

22. 最长公共子序列

## 前缀和
845. 数组中的最长山脉
1310. 子数组异或查询. 

## 数学推导
11. 盛水最多的容器
48. 旋转图像
66. 加一
70. 爬楼梯
119. 杨辉三角II 
	- 排列计算公式
136. 只出现一次的数字
179. 最大数
268. 丢失的数字
292. Nim游戏
409. 最长回文串
566. 重塑矩阵
766. 托普利茨矩阵
888. 公平的糖果交换
1217. 玩筹码
5677. 统计同构子字符串的数目

## 差分数列

1109. 航班预计统计

## 并查集

547. 省份数量
684. 冗余链接
765. 情侣牵手

## 位运算
136. 只出现一次的数字
137. 只出现一次的数字 II
260. 只出现一次的数字 III
645. 错误的集合

## 巧解
0. 面经. 排序123数组
134. 加油站
136. 只出现一次的数字
169. 多数元素 （摩尔投票法）
283. 移动零
406. 根据身高重建队列
442. 数组中重复的数据
448. 找到所有数组中消失的数字
453. 最小操作次数使数组元素相等
459. 重复的子字符串
485. 最大连续1的个数
820. 单词编码
1128. 等价多米诺骨牌的数量
5673. 移除石子的最大得分

## 实现接口
208. 实现 trie 前缀树
232. 用栈实现队列
303. 区域和检索 - 数组不可变
304. 二维区域和检索 - 矩阵不可变
341. 扁平化嵌套列表迭代器

## 其他
4. 寻找两个正序数组的中位数
6. Z字形变换
12. 整数转罗马数字
14. 最长公共前缀
28. 实现strStr
38. 外观数列
41. 缺失的第一个正数
43. 字符串相乘
56. 合并区间
73. 矩阵置零
122. 买卖股票的最佳时机II
134. 加油站
165. 比较版本号
169. 多数元素
189. 旋转数组
228. 汇总区间
238. 除自身以外的数组的乘积
242. 有效的字母异位词
253. 会议室II
287. 寻找重复数
347. 前K个高频元素
392. 判断子序列
456. 132模式（用到了multiset）
463. 岛屿的周长
485. 最大的连续1的个数
554. 砖墙
561. 数组拆分I
665. 非递减数列
704. 二分查找
781. 森林中的兔子
784. 字母大小写全排列
821. 字符的最短距离
832. 反转图像
867. 转置矩阵
912. 排序数组
918. 环形子数组的最大和
945. 使数组唯一的最小增量
1323. 6和9组成的最大数字
1429. 第一个唯一的数字
1310. 子数组异或查询（前缀和）
1720. 解码异或后的数组
1734. 解码异或后的排列
5672. 检查数组是否经排序和轮转得到
5674. 构造字典序最大的合并字符串
5676. 生成交替二进制字符串的最少操作数



