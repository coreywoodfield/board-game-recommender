using CSV, DataFrames, Query, Missings, Gadfly


data = CSV.read("movie_training_data/user_ratedmovies_train.dat", delim='\t', nullable=false)

userids = unique(data[:userID])
movieids = unique(data[:movieID])

userdict = Dict(id => index for (index, id) in enumerate(userids))
moviedict = Dict(id => index for (index, id) in enumerate(movieids))

# add new columns with more useful ids
data[:user] = [userdict[id] for id in data[:userID]]
data[:movie] = [moviedict[id] for id in data[:movieID]]

rows = @from row in data begin
	@select {user=row.user[], movie=row.movie[], row.rating}
	@collect
end

shuffle!(rows)
test_rows = rows[1:85000]
train_rows = rows[85001:end]

function make_ratings_matrix(rows)
	ratings = missings(Float64, length(userids), length(movieids))
	foreach(row -> ratings[row.user, row.movie] = row.rating, rows)
	ratings
end

ratings = make_ratings_matrix(train_rows)
v_ratings = make_ratings_matrix(test_rows)

K = 200
λ = 0.02
# decaying learning rate
ηt = [.01,.01,.01,.01,.005,.005,.005,.005,.001,.001,.0001,.0001]

users = rand(length(userids), K)
movies = rand(K, length(movieids))

function rse(ratings, predictions)
	err = mapreduce(x->x^2, +, 0, skipmissing(ratings - predictions))
	err + λ * (vecnorm(users)^2 + vecnorm(movies)^2)
end

function rmse(ratings, predictions)
	diffs = collect(skipmissing(ratings - predictions))
	n = length(diffs)
	sqrt(mapreduce(x->x^2, +, 0, diffs) / n)
end

predictions = users * movies
train_rse = [rse(ratings, predictions)]
test_rse = [rse(v_ratings, predictions)]
train_rmse = [rmse(ratings, predictions)]
test_rmse = [rmse(ratings, predictions)]

function update(rating, user, movie, η)
	predicted = user ⋅ movie
	err = rating - predicted
	Δu = η * (err * movie - λ * user)
	Δm = η * (err * user - λ * movie)
	user + Δu, movie + Δm
end

for η in ηt
	shuffle!(train_rows)
	for row in train_rows
		u, m = update(row.rating, users[row.user, :], movies[:, row.movie], η)
		users[row.user, :] .= u
		movies[:, row.movie] .= m
	end
	predictions = users * movies
	push!(train_rse, rse(ratings, predictions))
	push!(test_rse, rse(v_ratings, predictions))
	push!(train_rmse, rmse(ratings, predictions))
	push!(test_rmse, rmse(v_ratings, predictions))
end

plot_data1 = stack(DataFrame(x=0:12,Train=train_rse,Test=test_rse), [:Train, :Test])
plot_data2 = stack(DataFrame(x=1:12,Train=train_rse[2:end],Test=test_rse[2:end]), [:Train, :Test])
graph1 = plot(plot_data1, x=:x, y=:value, color=:variable, Geom.line, Geom.point, Scale.y_log10,
			  Guide.xlabel("Epochs"), Guide.ylabel("RSE"), Guide.title("RSE including initial state"))
graph2 = plot(plot_data2, x=:x, y=:value, color=:variable, Geom.line, Geom.point,
			  Guide.xlabel("Epochs"), Guide.ylabel("RSE"), Guide.title("RSE excluding initial state"))
draw(PNG("RSE1.png", 6inch, 6inch/golden), graph1)
draw(PNG("RSE2.png", 6inch, 6inch/golden), graph2)
plot_data1 = stack(DataFrame(x=0:12,Train=train_rmse,Test=test_rmse), [:Train, :Test])
plot_data2 = stack(DataFrame(x=1:12,Train=train_rmse[2:end],Test=test_rmse[2:end]), [:Train, :Test])
graph1 = plot(plot_data1, x=:x, y=:value, color=:variable, Geom.line, Geom.point,
			  Guide.xlabel("Epochs"), Guide.ylabel("RMSE"), Guide.title("RMSE including initial state"))
graph2 = plot(plot_data2, x=:x, y=:value, color=:variable, Geom.line, Geom.point,
			  Guide.xlabel("Epochs"), Guide.ylabel("RMSE"), Guide.title("RMSE excluding initial state"))
draw(PNG("RMSE1.png", 6inch, 6inch/golden), graph1)
draw(PNG("RMSE2.png", 6inch, 6inch/golden), graph2)

function getrating(userid, movieid)
	if haskey(userdict, userid) && haskey(moviedict, movieid)
		prediction = predictions[userdict[userid],moviedict[movieid]]
		prediction > 5.0 ? 5.0 : prediction < 0.5 ? 0.5 : prediction
	elseif haskey(userdict, userid)
		mean(skipmissing(ratings[userdict[userid],:]))
	elseif haskey(moviedict, movieid)
		mean(skipmissing(ratings[:,moviedict[movieid]]))
	end
end

test_stuff = CSV.read("predictions.dat", delim="\t", nullable=false)
output = @from row in test_stuff begin
	@let predicted = getrating(row.userID, row.movieID)
	@select {row.testID, predicted_rating = predicted}
	@collect
end
writecsv("submission.csv", output)
