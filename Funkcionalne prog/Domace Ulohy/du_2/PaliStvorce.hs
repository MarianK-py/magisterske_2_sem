module PaliStvorce where

-- Marian Kravec

parPocetCif :: Integer -> Bool
parPocetCif n | n < 10 = False
              | otherwise = not $ parPocetCif (n `div` 10)

parPocetCif' :: Integer -> Bool
parPocetCif' n = ((floor (logBase 10 (fromIntegral n))) `mod` 2) == 1

palindrom :: String -> Bool
palindrom [] = True
palindrom (s:[]) = True
palindrom s = ((head s) == (last s)) && (palindrom (init $ tail s))

palindrom' :: String -> Bool
palindrom' s = s == reverse s

palindrom'' :: Integer -> Bool
palindrom'' n = let str = show n
                 in str == reverse str


nte :: Int -> Integer
nte i = test 32 1 2
        where
          test :: Integer -> Int -> Integer -> Integer
          test n k j | n < (10^j) = if palindrom' $ show (n^2)
                                         then if k == i
                                              then n^2
                                              else test (n+1) (k+1) j
                                         else test (n+1) k j
                     | otherwise = test (ceiling ((fromInteger n)*3.1622776601683795)) k (j+1)



sqrs :: [Integer]
sqrs = numgen 32
       where
         numgen :: Integer -> [Integer]
         numgen n = (n^2):(numgen (n+1))

pwr10 :: Integer -> Bool
pwr10 y = y == (10^((floor $ logBase 10 (fromInteger (y+1)))))

sqrs' :: [Integer]
sqrs' = numgen 32
       where
         numgen :: Integer -> [Integer]
         numgen n | not $ pwr10 n = (n^2):(numgen (n+1))
                  | otherwise = numgen (ceiling ((fromInteger n)*3.1622776601683795))

sqrs'' :: [Integer]
sqrs'' = numgen 1024 65
       where
         numgen :: Integer -> Integer -> [Integer]
         numgen n k = n:(numgen (n+k) (k+2))

nte' :: Int -> Integer
nte' i = (filter palindrom'' (filter parPocetCif' sqrs))!!(i-1)

nte'' :: Int -> Integer
nte'' i = (filter palindrom'' sqrs')!!(i-1)

nte''' :: Int -> Integer
nte''' i = (filter palindrom'' (filter parPocetCif' sqrs''))!!(i-1)
