import Control.Monad
import Data.List

{-
kombinacie [(+)] [1,2] [3,4] = nub [1+3, 1+4, 2+3, 2+4] = [4,5,6]
kombinacie [1,2] [3,4] [5,6] = nub [1+3+5, 1+4+5, 2+3+5, 2+4+5, 1+3+6, 1+4+6, 2+3+6, 2+4+6] = [9,10,11,12]
kombinacie [(+), (subtract)] [1,2] [3,4] [5,6] =
  nub [1+3+5, 1+4+5, 2+3+5, 2+4+5, 1+3+6, 1+4+6, 2+3+6, 2+4+6,
       1+3-5, 1+4-5, 2+3-5, 2+4-5, 1+3-6, 1+4-6, 2+3-6, 2+4-6,
       1-3+5, 1-4+5, 2-3+5, 2-4+5, 1-3+6, 1-4+6, 2-3+6, 2-4+6,
       1-3-5, 1-4-5, 2-3-5, 2-4-5, 1-3-6, 1-4-6, 2-3-6, 2-4-6] = [9,10,11,12,-1,0,1,-2,3,2,4,5,-7,-8,-6,-9]
-}

kombinacie :: Eq a => [a->a->a] -> [[a]] -> [a]
kombinacie funk (xs:ys:[]) =  let
                                pairs = (pure (,) <*> xs <*> ys)
                              in
                                nub (concat ([map (\(x, y) -> f x y) pairs | f <- funk]))
kombinacie funk (xs:xss) = kombinacie funk [xs, (kombinacie funk xss)]

main :: IO ()
main = do
        putStrLn $ show $ kombinacie [(+),(subtract), (*), div] [[0,1],[1,2,3]]