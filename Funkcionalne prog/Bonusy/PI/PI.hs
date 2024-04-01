module SI where

import Prelude hiding (pi)

-- Marian Kravec

pi :: [Int]
pi = [(fromEnum (piii*(10^i))) `mod` 10 | i <- [0..1000]]
     where
       piii = pii 100

a :: [Double]
a = 1:[((a!!i) + (b!!i))/2 | i <- [0..]]

b :: [Double]
b = (1/(sqrt 2)):[sqrt ((b!!i)*(a!!i)) | i <- [0..]]

t :: [Double]
t = (1/4):[(t!!i) - ((p!!i)*(((a!!i)-(a!!(i+1)))^2)) | i <- [0..]]

p :: [Double]
p = 1:[2*(p!!i) | i <- [0..]]

pii :: Int -> Double
pii i = (((a!!i)+(b!!i))^2)/(4*(t!!i))