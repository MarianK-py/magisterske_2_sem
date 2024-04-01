module FLab where

-- Marian Kravec

type  Lab = [[Int -> Int]]
findMaxPath          :: Lab -> Int -> Int
findMaxPath lab init = maxPath 0 0 [] init
                       where
                         m = (length lab)-1
                         n = (length (lab!!0))-1

                         maxPath i j visited curr | i < 0 = (minBound :: Int)
                                                  | j < 0 = (minBound :: Int)
                                                  | i > m = (minBound :: Int)
                                                  | j > n = (minBound :: Int)
                                                  | elem (i, j) visited = if (length visited) == ((m+1)*(n+1))
                                                                          then curr
                                                                          else (minBound :: Int)
                                                  | otherwise = lisMax [maxPath (i+1) j ((i, j):visited) ((lab!!i!!j) curr), maxPath (i-1) j ((i, j):visited) ((lab!!i!!j) curr), maxPath i (j+1) ((i, j):visited) ((lab!!i!!j) curr), maxPath i (j-1) ((i, j):visited) ((lab!!i!!j) curr)]


lisMax :: [Int] -> Int
lisMax xs = lm' xs (minBound :: Int)
            where
              lm' [] m = m
              lm' (x:xs) m | x > m = lm' xs x
                           | otherwise = lm' xs m

lab1 :: Lab
lab1  = [
           [ (+1), (+1),(+1) ],
           [ (+1), (+1),(+1) ],
           [ (+1), (*2),(+1) ]
        ]

lab2  :: Lab
lab2  = [
          [ (+1), (+1),(+1) ],
          [ (+1), (+1),(*2) ],
          [ (+1), (*2),(+1) ]
        ]
-- findMaxPath lab2 0 = 23

lab3  :: Lab
lab3  = [
          [ (+1), (+1),(+1) ],
          [ (*2), (+1),(*2) ],
          [ (+1), (*2),(+1) ]
        ]
-- findMaxPath lab3 0 = 31

lab4  :: Lab
lab4  = [
          [ (+1), (*2),(*3) ],
          [ (*2), (`div` 3),(*2) ],
          [ (+4), (*2),(+10) ]
        ]
-- findMaxPath lab4 0 = 108

lab5  :: Lab
lab5  = [
          [ (*2), (*3),(*5) ],
          [ (*8), (*9),(*4) ],
          [ (*7), (*6),(*10) ]
        ]
