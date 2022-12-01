//jshint esversion:6

require('dotenv').config();
currentYear = new Date().getFullYear();
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
complete_answer=""
// knn
knn_bin_cls=""
knn_mul_cls=""
knn_desc=""
knn_bin_acc="0.9760368900303525"
knn_mul_acc="0.9740368900303525"
// random forest
rf_bin_cls=""
rf_mul_cls=""
rf_desc=""
rf_bin_acc="0.9741029652113005"
rf_mul_acc="0.9731029652113005"
// cnn
cnn_bin_cls=""
cnn_mul_cls=""
cnn_desc=""
cnn_bin_acc="0.8248890964277376"
cnn_mul_acc="0.772682699042727"
//lstm
lstm_bin_cls=""
lstm_mul_cls=""
lstm_desc=""
lstm_bin_acc="0.828017744571562"
lstm_mul_acc="0.7606350688769554"
app.get("/", function(req, res){
  res.render("home");
  let options={
    args:[]
  };
  PythonShell.run('nids_parameter.py',options, (err,response)=>{
    if (err)
    console.log(err);
    if(response){
      complete_answer=stringify(response);

      //knn
      temp_knn_bin_cls=stringify(response[1]);
      knn_bin_cls=temp_knn_bin_cls.slice(2,-2);

      temp_knn_mul_cls=stringify(response[2]);
      knn_mul_cls=temp_knn_mul_cls.slice(2,-2);

      temp_knn_desc=stringify(response[3]);
      knn_desc=temp_knn_desc.slice(2,-2);
      //random forest
      temp_rf_bin_cls=stringify(response[5]);
      rf_bin_cls=temp_rf_bin_cls.slice(2,-2);

      temp_rf_mul_cls=stringify(response[6]);
      rf_mul_cls=temp_rf_mul_cls.slice(2,-2);

      temp_rf_desc=stringify(response[7]);
      rf_desc=temp_rf_desc.slice(2,-2);
      //cnn
      temp_cnn_bin_cls=stringify(response[9]);
      cnn_bin_cls=temp_cnn_bin_cls.slice(2,-2);

      temp_cnn_mul_cls=stringify(response[10]);
      cnn_mul_cls=temp_cnn_mul_cls.slice(2,-2);

      temp_cnn_desc=stringify(response[11]);
      cnn_desc=temp_cnn_desc.slice(2,-2);
      //lstm
      temp_lstm_bin_cls=stringify(response[13]);
      lstm_bin_cls=temp_lstm_bin_cls.slice(2,-2);

      temp_lstm_mul_cls=stringify(response[14]);
      lstm_mul_cls=temp_lstm_mul_cls.slice(2,-2);

      temp_lstm_desc=stringify(response[15]);
      lstm_desc=temp_lstm_desc.slice(2,-2);
    }
  });
});

app.get("/secrets",function(req,res){
  res.render("secrets");
})

complete_answer=""
// knn
knn_bin_cls=""
knn_mul_cls=""
knn_desc=""
knn_bin_acc="0.9760368900303525"
knn_mul_acc="0.9740368900303525"
// random forest
rf_bin_cls=""
rf_mul_cls=""
rf_desc=""
rf_bin_acc="0.9741029652113005"
rf_mul_acc="0.9731029652113005"
// cnn
cnn_bin_cls=""
cnn_mul_cls=""
cnn_desc=""
cnn_bin_acc="0.8248890964277376"
cnn_mul_acc="0.772682699042727"
//lstm
lstm_bin_cls=""
lstm_mul_cls=""
lstm_desc=""
lstm_bin_acc="0.828017744571562"
lstm_mul_acc="0.7606350688769554"
app.post("/parameters",function(req,res)
{
  const submitted_protocol_type=req.body.protocol_type;
  const submitted_service=req.body.service;
  const submitted_flag=req.body.flag;
  const submitted_logged_in=req.body.logged_in;
  const submitted_count=req.body.count;
  const submitted_srv_serror_rate=req.body.srv_serror_rate;
  const submitted_srv_rerror_rate=req.body.srv_rerror_rate;
  const submitted_same_srv_rate=req.body.same_srv_rate;
  const submitted_diff_srv_rate=req.body.diff_srv_rate;
  const submitted_dst_host_count=req.body.dst_host_count;
  const submitted_dst_host_srv_count=req.body.dst_host_srv_count;
  const submitted_dst_host_same_srv_rate=req.body.dst_host_same_srv_rate;
  const submitted_dst_host_diff_srv_rate=req.body.dst_host_diff_srv_rate;
  const submitted_dst_host_same_src_port_rate=req.body.dst_host_same_src_port_rate;
  const submitted_dst_host_serror_rate=req.body.dst_host_serror_rate;
  const submitted_dst_host_rerror_rate=req.body.dst_host_rerror_rate;

  let options={
    args:[submitted_protocol_type,submitted_service,submitted_flag,submitted_logged_in,submitted_count,submitted_srv_serror_rate,submitted_srv_rerror_rate,submitted_same_srv_rate,submitted_diff_srv_rate,submitted_dst_host_count,submitted_dst_host_srv_count,submitted_dst_host_same_srv_rate,submitted_dst_host_diff_srv_rate,submitted_dst_host_same_src_port_rate,submitted_dst_host_serror_rate,submitted_dst_host_rerror_rate]
  };
  PythonShell.run('tp.py',options, (err,response)=>{
    if (err)
    console.log(err);
    if(response){
      complete_answer=stringify(response);

      //knn
      temp_knn_bin_cls=stringify(response[1]);
      knn_bin_cls=temp_knn_bin_cls.slice(2,-2);

      temp_knn_mul_cls=stringify(response[2]);
      knn_mul_cls=temp_knn_mul_cls.slice(2,-2);

      temp_knn_desc=stringify(response[3]);
      knn_desc=temp_knn_desc.slice(2,-2);
      //random forest
      temp_rf_bin_cls=stringify(response[5]);
      rf_bin_cls=temp_rf_bin_cls.slice(2,-2);

      temp_rf_mul_cls=stringify(response[6]);
      rf_mul_cls=temp_rf_mul_cls.slice(2,-2);

      temp_rf_desc=stringify(response[7]);
      rf_desc=temp_rf_desc.slice(2,-2);
      //cnn
      temp_cnn_bin_cls=stringify(response[9]);
      cnn_bin_cls=temp_cnn_bin_cls.slice(2,-2);

      temp_cnn_mul_cls=stringify(response[10]);
      cnn_mul_cls=temp_cnn_mul_cls.slice(2,-2);

      temp_cnn_desc=stringify(response[11]);
      cnn_desc=temp_cnn_desc.slice(2,-2);
      //lstm
      temp_lstm_bin_cls=stringify(response[13]);
      lstm_bin_cls=temp_lstm_bin_cls.slice(2,-2);

      temp_lstm_mul_cls=stringify(response[14]);
      lstm_mul_cls=temp_lstm_mul_cls.slice(2,-2);

      temp_lstm_desc=stringify(response[15]);
      lstm_desc=temp_lstm_desc.slice(2,-2);
    }
  });
  res.redirect("/paramsecrets");
})

app.get("/features",function(req,res){
  res.render("features");
})
app.get("/attacks",function(req,res){
  res.render("attacks");
})
app.get("/about",function(req,res){
  res.render("about");
})
app.get("/table",function(req,res){
  res.render("table");
})
app.get("/parameters",function(req,res){
  res.render("parameters");
})
app.get("/contact",function(req,res){
  res.render("contact");
})
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
