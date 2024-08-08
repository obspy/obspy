"use strict";

// load plugins
const autoprefixer = require("gulp-autoprefixer");
const gulp = require('gulp');
const uglify = require('gulp-uglify-es').default;
const del = require('delete');
const rename = require('gulp-rename');
const js_import = require('gulp-js-import');
const babel = require('gulp-babel');
const clean_css = require('gulp-clean-css');
const imagemin = require('gulp-imagemin');
const sass = require('gulp-sass');
const sass_lint = require('gulp-sass-lint');
const csso = require('gulp-csso');


// define paths
const PATHS = {
  scripts: {
    src: './source/_static/js/*.js',
    dest: './source/_static/js/',
    dest2: './build/html/_static/js/',
  },
  styles: {
    src: './source/_static/css/*.scss',
    dest: './source/_static/css/',
    dest2: './build/html/_static/css/',
  },
  images: {
    src: './source/_images/**/*',
    dest: './source/_images/',
  },
  logos: {
    src: './source/_static/*logo*',
    dest: './source/_static/',
  },
};


// optimise images
function optimise_images() {
  return gulp
    .src(PATHS.images.src)
    .pipe(imagemin({'verbose': true}))
    .pipe(gulp.dest(PATHS.images.dest));
}

function optimise_logos() {
  return gulp
    .src(PATHS.logos.src)
    .pipe(imagemin({'verbose': true}))
    .pipe(gulp.dest(PATHS.logos.dest));
}

const images = gulp.parallel(
    optimise_images,
    optimise_logos,
);


// handle scripts
function scripts_clean() {
  return del([PATHS.scripts.dest + '*.min.js',
              PATHS.scripts.dest + 'maps/*.map']);
}

function scripts_build() {
  return gulp
    .src(PATHS.scripts.src, { sourcemaps: true })
    .pipe(js_import({hideConsole: true}))
    .pipe(babel())
    .pipe(uglify())
    .pipe(rename({extname: '.min.js'}))
    .pipe(gulp.dest(PATHS.scripts.dest, { sourcemaps: './maps' }))
    .pipe(gulp.dest(PATHS.scripts.dest2, { sourcemaps: './maps' }));
}

const js = gulp.series(scripts_clean, scripts_build);


// handle styles
function styles_clean() {
  return del([PATHS.styles.dest + '*.min.css',
              PATHS.styles.dest + '**/*.map']);
}

function styles_check() {
  var sass_lint_options = {
    rules: {
      'property-sort-order': 0,
      'no-ids': 0,
      'nesting-depth': 0
    }
  }
  return gulp
    .src(PATHS.styles.src)
    .pipe(sass_lint(sass_lint_options))
    .pipe(sass_lint.format())
    .pipe(sass_lint.failOnError());
}

function styles_build() {
  return gulp
    .src(PATHS.styles.src, { sourcemaps: true })
    .pipe(sass({outputStyle : 'expanded'}))
    .pipe(autoprefixer(['last 3 versions', '> 1%'], {cascade : true}))
    .pipe(csso())
    .pipe(clean_css())
    .pipe(rename({extname: '.min.css'}))
    .pipe(gulp.dest(PATHS.styles.dest, { sourcemaps: './maps' }))
    .pipe(gulp.dest(PATHS.styles.dest2, { sourcemaps: './maps' }));
}

const css = gulp.series(styles_check, styles_clean, styles_build);


// define complex task
function watch_files() {
  gulp.watch(PATHS.styles.src, gulp.series(css));
  gulp.watch(PATHS.scripts.src, gulp.series(js));
}

const watch = gulp.parallel(watch_files);
const build = gulp.parallel(css, js);
const clean = gulp.parallel(scripts_clean, styles_clean);


// export tasks
exports.build = build;
exports.watch = watch;
exports.images = images;
exports.css = css;
exports.js = js;
exports.clean = clean;
exports.default = watch;
