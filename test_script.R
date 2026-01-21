devtools::load_all()

#add a copy of the old column with new name
mtcars |> 
  tbl_gpu() |> 
  mutate(am_2 = am) 

#get all
mtcars |> 
  tbl_gpu() |> 
  filter(rep(TRUE, nrow(mtcars)))

#get all
mtcars |> 
  tbl_gpu() |> 
  filter(TRUE)

#get none
mtcars |> 
  tbl_gpu() |> 
  filter(rep(FALSE, nrow(mtcars)))

#get none
mtcars |> 
  tbl_gpu() |> 
  filter(FALSE)

mtcars |>
    tbl_gpu() |>
    group_by(vs) |>
    summarise(
      n = n(),
      min_am = max(am*100),
      n_carb_4 = sum(carb == 4)
    ) |>
    collect()
