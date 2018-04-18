// The file contents for the current environment will overwrite these during build.
// The build system defaults to the dev environment which uses `environment.ts`, but if you do
// `ng build --env=prod` then `environment.prod.ts` will be used instead.
// The list of which env maps to which file can be found in `.angular-cli.json`.

// src/environments/environment.ts
// used when we run `ng serve` or `ng build`
export const environment = {
  production: false,

  // URL of development API
  apiUrl: 'http://192.168.88.228:8000/',

  linkedInApiUrl: 'https://www.linkedin.com/voyager/api/'
};
