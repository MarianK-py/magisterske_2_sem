module DropIndex where

-- Marian Kravec

drop'       :: Int -> [t] -> [t]
drop' n xs  =  let
                 (a, b) = foldl (\(z, i) x -> if i < n then (z, (i+1)) else (z++[x], (i+1))) ([], 0) xs
               in
                 a

(!!!!)      :: [t] -> Int -> t
xs !!!! n   =  let
                 (a, b) = foldl (\(z, i) x -> if i == n then (x, (i+1)) else (z, (i+1))) (head xs, 0) xs
               in
                 a



-- naprogramujme take pomocou foldl/r

take' :: Int -> [a] -> [a]
take' n xs  =  (foldr pom (\_ -> []) xs) n where
                pom x h = \n -> if n == 0 then []
                                 else x:(h (n-1))

take'' :: Int -> [a] -> [a]
take'' n xs  =  (foldr pom (\_ -> []) xs) n where
                  pom x h n = if n == 0 then []
                              else x:(h (n-1))

take''' n xs = foldr (\a h -> \n -> case n of
                                         0 -> []
                                         n -> a:(h (n-1)) )
                     (\_ -> [])
               xs
               n