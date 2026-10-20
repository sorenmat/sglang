# GPU host policy

Standing rule from sorenmat (2026-10-19): whenever a cloud GPU instance
is provisioned for these experiments, it MUST be torn down when the
measurement session ends — no idle boxes left burning spot dollars.

For GCP (`kapeta-staging`):

    gcloud compute instances delete INSTANCE \
      --zone ZONE --project kapeta-staging --quiet

The boot disk is created with auto-delete=yes, so deleting the instance
removes it. Detached extra disks, if any were added, must be deleted
separately by name.

SSH access on fresh instances: generate a passphrase-less key and add it
to the INSTANCE metadata (project-wide keys belong to other people's
laptops; the shared gcloud key on this workstation is passphrase
protected and cannot be used non-interactively):

    ssh-keygen -t ed25519 -N "" -f ~/.ssh/hkv6_box
    gcloud compute instances add-metadata INSTANCE --zone ZONE \
      --project kapeta-staging \
      --metadata="ssh-keys=runner:$(cat ~/.ssh/hkv6_box.pub)"
    ssh -i ~/.ssh/hkv6_box -o IdentitiesOnly=yes runner@IP

Account: `soren.mathiasen@neuralogics.com` (has compute access to
kapeta-staging; `sorenmat@gmail.com` does not).
