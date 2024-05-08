module DirList where

import Control.Monad
import System.Directory
import System.Environment

-- System.Directory
-- doesDirectoryExist has type FilePath -> IO Bool

vizualizeDir :: FilePath -> String -> String -> IO ()
vizualizeDir p s ind = do
                         putStrLn (ind++"->"++s)
                         subPathExistence <- doesDirectoryExist (p++"\\"++s)
                         if subPathExistence
                         then
                           do
                             subPaths <-  listDirectory (p++"\\"++s)
                             mapM_ (\m -> vizualizeDir (p++"\\"++s) m ("  "++ind)) subPaths
                         else
                           return ()

main :: IO ()
main = do dir <- getArgs
          mapM_ (\m -> vizualizeDir m "" "") dir
          --filesOrDirs <- mapM listDirectory dir
          --putStrLn $ show filesOrDirs