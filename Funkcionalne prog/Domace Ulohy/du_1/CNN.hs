module CNN where

-- Marian Kravec

cnn :: Int
cnn = finder 10
      where
        finder i | (prime i) && (prime (reverseNum i)) && (binPalin i) && (binPalin (reverseNum i)) = i
                 | otherwise = finder (i+1)


reverseNum :: Int -> Int
reverseNum n = accum n 0
               where
                 accum m acc | m == 0 = acc
                             | otherwise = accum (m `div` 10) ((10*acc)+(m `mod` 10))

prime :: Int -> Bool
prime n = primeTest 2
          where
            primeTest :: Int -> Bool
            primeTest i | i < n = if (n `mod` i) == 0 then False else primeTest (i+1)
                        | otherwise = True

toBin :: Int -> [Int]
toBin n = accum n []
          where
            accum :: Int -> [Int] -> [Int]
            accum n xs | n > 0 = accum (n `div` 2) ((n `mod` 2):xs)
                       | otherwise = xs

binPalin :: Int -> Bool
binPalin n = (toBin n) == (reverse (toBin n))