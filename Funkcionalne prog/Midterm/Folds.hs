module Folds where

-- Marian Kravec

geomR :: [Float]->Float
geomR xs = fst (foldl (\(acc, k) x -> ((x*(acc**k))**(1/(k+1)), k+1)) (1, 0) xs)

geomM :: [[Float]]->Float
geomM xs = fst (foldl (\(acc, k) x -> ((((geomR x)**(fromIntegral (length x)))*(acc**k))**(1/(k+(fromIntegral (length x)))), k+(fromIntegral (length x)))) (1,0) xs)

-- pomocna funkcia, snad nie je zakazana :P
pairDiv :: (Float, Float) -> Float
pairDiv (a, b) = a/b

harmoR :: [Float]->Float
harmoR xs = pairDiv (foldr (\x (k, acc) -> (k+1.0, acc+(1/x))) (0.0, 0.0) xs)

harmoM :: [[Float]]->Float
harmoM xs = pairDiv (foldr (\x (k, acc) -> (k+(fromIntegral (length x)), acc+((fromIntegral (length x))*(1.0/(harmoR x))))) (0.0, 0.0) xs)