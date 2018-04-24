using CSV, DataFrames, Query, Gadfly, GZip

io = gzopen("games.csv.gz")
gamelist = CSV.read(io, nullable=false)
close(io)
data = CSV.read("ratings.csv", nullable=false)

usernames = unique(data[:username])
usercount = length(usernames)
gamecount = size(gamelist)[1]
userdict = Dict(username => index for (index, username) in enumerate(usernames))
gamedict = Dict(id => index for (index, id) in enumerate(gamelist[:id]))
gamenames = Dict(id => name for (id, name) in zip(gamelist[:id], gamelist[:name]))

# add new columns with more useful ids
data[:uidx] = [userdict[username] for username in data[:username]]
data[:gidx] = [gamedict[id] for id in data[:gameid]]

rows = @from row in data begin
	@select [row.uidx[], row.gidx[], row.rating]
	@collect
end

shuffle!(rows)
test = floor(Int, .1 * length(rows))
test_rows = rows[1:test]
train_rows = rows[test+1:end]

function make_ratings_matrix(rows)
	matrix = hcat(rows...)
	sparse(matrix[1,:], matrix[2,:], matrix[3,:], usercount, gamecount)
end

ratings = make_ratings_matrix(train_rows)
v_ratings = make_ratings_matrix(test_rows)

K = 10
λ = 0.02
# decaying learning rate
ηt = [.01,.01,.01,.01]

users = rand(usercount, K)
games = rand(K, gamecount)

get_diff(user, game, actual, predictions) = actual - predictions[user, game]

function rse(ratings, predictions)
	err = mapreduce(x->get_diff(x..., predictions)^2, +, 0, zip(findnz(ratings)...))
	err + λ * (vecnorm(users)^2 + vecnorm(games)^2)
end

function rmse(ratings, predictions)
	diffs = map(x->get_diff(x..., predictions), zip(findnz(ratings)...))
	n = length(diffs)
	sqrt(mapreduce(x->x^2, +, 0, diffs) / n)
end

predictions = users * games
train_rse = [rse(ratings, predictions)]
test_rse = [rse(v_ratings, predictions)]
train_rmse = [rmse(ratings, predictions)]
test_rmse = [rmse(v_ratings, predictions)]

function update(rating, user, game, η)
	predicted = user ⋅ game
	err = rating - predicted
	Δu = η * (err * game - λ * user)
	Δg = η * (err * user - λ * game)
	user + Δu, game + Δg
end

for η in ηt
	gc()
	shuffle!(train_rows)
	for row in train_rows
		user, game, rating = row
		user = Int(user)
		game = Int(game)
		u, g = update(rating, users[user, :], games[:, game], η)
		users[user, :] .= u
		games[:, game] .= g
	end
	predictions = users * games
	push!(train_rse, rse(ratings, predictions))
	push!(test_rse, rse(v_ratings, predictions))
	push!(train_rmse, rmse(ratings, predictions))
	push!(test_rmse, rmse(v_ratings, predictions))
	println(η, " done")
end

plot_data1 = stack(DataFrame(x=0:4,Train=train_rse,Test=test_rse), [:Train, :Test])
plot_data2 = stack(DataFrame(x=1:4,Train=train_rse[2:end],Test=test_rse[2:end]), [:Train, :Test])
graph1 = plot(plot_data1, x=:x, y=:value, color=:variable, Geom.line, Geom.point, Scale.y_log10,
			  Guide.xlabel("Epochs"), Guide.ylabel("RSE"), Guide.title("RSE including initial state"))
graph2 = plot(plot_data2, x=:x, y=:value, color=:variable, Geom.line, Geom.point,
			  Guide.xlabel("Epochs"), Guide.ylabel("RSE"), Guide.title("RSE excluding initial state"))
draw(PNG("RSE1.png", 6inch, 6inch/golden), graph1)
draw(PNG("RSE2.png", 6inch, 6inch/golden), graph2)
plot_data1 = stack(DataFrame(x=0:4,Train=train_rmse,Test=test_rmse), [:Train, :Test])
plot_data2 = stack(DataFrame(x=1:4,Train=train_rmse[2:end],Test=test_rmse[2:end]), [:Train, :Test])
graph1 = plot(plot_data1, x=:x, y=:value, color=:variable, Geom.line, Geom.point,
			  Guide.xlabel("Epochs"), Guide.ylabel("RMSE"), Guide.title("RMSE including initial state"))
graph2 = plot(plot_data2, x=:x, y=:value, color=:variable, Geom.line, Geom.point,
			  Guide.xlabel("Epochs"), Guide.ylabel("RMSE"), Guide.title("RMSE excluding initial state"))
draw(PNG("RMSE1.png", 6inch, 6inch/golden), graph1)
draw(PNG("RMSE2.png", 6inch, 6inch/golden), graph2)

function getrating(userid, gameid)
	# if haskey(userdict, userid) && haskey(gamedict, gameid)
		prediction = predictions[userdict[userid],gamedict[gameid]]
		prediction > 10.0 ? 10.0 : prediction < 0.0 ? 0.0 : prediction
	# elseif haskey(userdict, userid)
		# mean(skipmissing(ratings[userdict[userid],:]))
	# elseif haskey(gamedict, gameid)
		# mean(skipmissing(ratings[:,gamedict[gameid]]))
	# end
end
