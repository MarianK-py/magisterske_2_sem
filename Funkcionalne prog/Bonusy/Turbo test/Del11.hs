module Del11 where

-- Marian Kravec

delitelne11 :: Integer -> Bool
delitelne11 0 = True
delitelne11 n | n <= 10 = False
              | otherwise = delitelne11 (rozdielCif n 0 True)

rozdielCif :: Integer -> Integer -> Bool -> Integer
rozdielCif n r s | n == 0 = abs r
                 | otherwise = if s == True
                               then
                                 rozdielCif (n `div` 10) (r + (n `mod` 10)) False
                               else
                                 rozdielCif (n `div` 10) (r - (n `mod` 10)) True
