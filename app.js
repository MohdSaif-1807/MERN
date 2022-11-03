//jshint esversion:6
require('dotenv').config();
const {parse, stringify} = require('flatted');
let {PythonShell} = require('python-shell')
const express = require("express");
const bodyParser = require("body-parser");
const ejs = require("ejs");
const mongoose = require("mongoose");
const session = require('express-session');
const passport = require("passport");
const passportLocalMongoose = require("passport-local-mongoose");
var GoogleStrategy = require('passport-google-oauth20').Strategy;
const findOrCreate = require('mongoose-findorcreate');
const app = express();
app.use(express.static("public"));
app.set('view engine', 'ejs');
app.use(bodyParser.urlencoded({
  extended: true
}));
app.use(session({
  secret: "Our little secret.",
  resave: false,
  saveUninitialized: false
}));
app.use(passport.initialize());
app.use(passport.session());
mongoose.connect(process.env.DB_LINK, {useNewUrlParser: true});
mongoose.set("useCreateIndex", true);

const userSchema = new mongoose.Schema ({
  email: String,
  password: String,
  googleId:String,
});

userSchema.plugin(passportLocalMongoose);
userSchema.plugin(findOrCreate)

const User = new mongoose.model("User", userSchema);

passport.use(User.createStrategy());

passport.serializeUser(function(user,done)
{
    done(null,user.id);
});
passport.deserializeUser(function(id,done)
{
    User.findById(id,function(err,user)
    {
        done(err,user);
    });
});

passport.use(new GoogleStrategy({
  clientID: process.env.CLIENT_ID,
  clientSecret: process.env.CLIENT_SECRET,
  callbackURL: process.env.CALL_BACK_URL,
  userProfileUrl:   process.env.URL
},
function(accessToken, refreshToken, profile, cb) {
  User.findOrCreate({ googleId: profile.id,username:profile.id}, function (err, user) {
    return cb(err, user);
  });
}
));
bincls=""
mulcls=""
desc=""
binacc="0.9740835862713052"
mulacc="0.9725893065608219"
app.get("/", function(req, res){
  res.render("home");
  let options={
    args:[]
  };
  PythonShell.run('temp.py',options, (err,response)=>{
    if (err)
    console.log(err);
    if(response){
      bs=stringify(response[0]);
      bincls=bs.slice(2,-2);
      ms=stringify(response[1]);
      mulcls=ms.slice(2,-2);
      dc=stringify(response[2]);
      desc=dc.slice(2,-2);
    }
  });
});

app.get('/auth/google',
  passport.authenticate('google', { scope:
      ['profile' ] }
));
app.get("/auth/google/NIDS",
  passport.authenticate('google', { failureRedirect: "/login" }),
  function(req, res) {
    res.redirect("/submit");
  });
app.get("/login", function(req, res){
  res.render("login");
});
app.get("/secrets", function(req, res){
  res.render("secrets");
  let options={
    args:[]
  };
  PythonShell.run('temp.py',options, (err,response)=>{
    if (err)
    console.log(err);
    if(response){
      bs=stringify(response[0]);
      bincls=bs.slice(2,-2);
      ms=stringify(response[1]);
      mulcls=ms.slice(2,-2);
      dc=stringify(response[2]);
      desc=dc.slice(2,-2);
    }
  });
});
app.get("/register", function(req, res){
  res.render("register");
});
app.get("/submit",function(req,res)
{
  if (req.isAuthenticated()){
    res.render("submit");
  } else {
    res.redirect("/login");
  }
  
});

app.get("/logout", function(req, res){
  req.logout();
  res.redirect("/");
});

app.post("/register", function(req, res){
  User.register({username: req.body.username},req.body.password, function(err, user){
    if (err) {
      console.log(err);
      res.redirect("/register");
    } else {
      passport.authenticate("local")(req, res, function(){
        res.redirect("/submit");
      });
    }
  });
});

app.post("/login", function(req, res){

  const user = new User({
    username: req.body.username,
    password: req.body.password,
  });
  req.login(user, function(err){
    if (err) {
      console.log(err);
    } else {
      passport.authenticate("local")(req, res, function(){
        res.redirect("/submit");
      });
    }
  });
});

let port = process.env.PORT;
	if (port == null || port == "") {
  	port = 3000;
	}
app.listen(port, function() {
  console.log("Server started on port 3000.");
});
