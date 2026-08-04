# PiPlus BFM Asset Provenance

The PiPlus description was imported from the local `ht_urdf` checkout:

- package: `PiPlus_S_12L8A0G2H1W_LSE_260424`
- source root: `/localhdd/Projects/ht_urdf/ht_urdf/PiPlus_S_12L8A0G2H1W_LSE_260424`
- source MJCF SHA-256: `4e8e40d67014ba9f0ba95914345c29673cfa63eceb6c2685a597fff368b49072`
- source URDF SHA-256: `fe93a45d79520a9a7dadaf0ab3c145f7e613362e5272a98e5671a2defadaa0e5`

`xml/piplus_bfm.xml` is a training-specific derivative. Joint-level
`actuatorfrcrange` attributes were removed because MJLab creates configured DC
motor actuators in Python. A same-name motor was added for each of the 23 policy
joints so MotionLib and robot inspection retain an explicit action layout.
The root `<joint type="free">` was rewritten as the equivalent `<freejoint>`
element so MotionLib does not count the floating base as a 24th actuated body.

The URDF content and meshes are otherwise copied from the source package. The
vendored URDF uses LF line endings, so its byte hash differs from the CRLF
source even though its XML content is unchanged. Confirm the upstream asset
license before redistributing these assets outside the authorized project
environment.
