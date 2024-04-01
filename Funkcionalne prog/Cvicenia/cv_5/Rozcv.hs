module Rozcv where


-- Marian Kravec

maxRozdiel :: [Int] -> Int
maxRozdiel [] = -1
maxRozdiel (x:xs) = max a b
                    where
                      a = maxRozdiel xs
                      b = foldr (\y acc -> max (y-x) acc) (-1) xs

-- najefektivnejsie... fuha... neviem ci sa to da bez toho ze skontrolujem vsetky dvojice co je O(n^2)
-- co je rovnako efektivne ako moje (asi)...

-- najelegantnejsie v Haskelli... mne sa moje paci... nie je elegantne ale...
-- ja nemam vkus takze staci :P