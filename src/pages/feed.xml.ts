import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import { excerpt } from '../lib/excerpt';

export async function GET(context: { site: URL }) {
  const posts = (await getCollection('blog'))
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());

  return rss({
    title: 'Lucas Wiman',
    description: 'Lucas Wiman on programming, Python, concurrency testing, and mathematics.',
    site: context.site,
    items: posts.map((post) => ({
      title: post.data.title,
      pubDate: post.data.date,
      description: post.data.description ?? excerpt(post.body ?? ''),
      link: `/blog/${post.id}/`,
    })),
  });
}
